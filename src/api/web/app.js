"use strict";

function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Required interface element is missing: ${id}`);
  return element;
}

const form = requiredElement("prediction-form");
const submitButton = requiredElement("submit");
const submitLabel = requiredElement("submit-label");
const errorOutput = requiredElement("error");
const apiKey = requiredElement("api-key");
const toggleKey = requiredElement("toggle-key");
const refreshModel = requiredElement("refresh-model");
const resetForm = requiredElement("reset-form");
const serviceChip = requiredElement("service-chip");
const scoreProgress = requiredElement("score-progress");
const decisionBadge = requiredElement("decision-badge");
const importanceContainer = requiredElement("global-importance");
const internetService = form.elements.InternetService;
const phoneService = form.elements.PhoneService;
const multipleLines = form.elements.MultipleLines;
const tenure = form.elements.tenure;
const totalCharges = form.elements.TotalCharges;

function requestHeaders(includeJson = false) {
  const headers = {};
  if (includeJson) headers["Content-Type"] = "application/json";
  if (apiKey.value) headers["X-API-Key"] = apiKey.value;
  return headers;
}

function setText(id, value) {
  requiredElement(id).textContent = value;
}

function setServiceStatus(state, message) {
  serviceChip.dataset.state = state;
  setText("service-status", message);
}

function syncInternetServices() {
  const absent = internetService.value === "No";
  document.querySelectorAll("[data-internet-addon]").forEach((field) => {
    if (absent) field.value = "No internet service";
    else if (field.value === "No internet service") field.value = "No";
  });
}

function syncPhoneServices() {
  if (phoneService.value === "No") multipleLines.value = "No phone service";
  else if (multipleLines.value === "No phone service") multipleLines.value = "No";
}

function syncCharges() {
  const months = Number(tenure.value);
  const newAccount = months === 0;
  totalCharges.max = String(Math.max(0, months * 1000));
  if (newAccount) totalCharges.value = "0";
  totalCharges.readOnly = newAccount;
}

function payloadFromForm() {
  const values = new FormData(form);
  const payload = Object.fromEntries(values.entries());
  if (!payload.customer_id.trim()) delete payload.customer_id;
  payload.SeniorCitizen = Number(payload.SeniorCitizen);
  payload.tenure = Number(payload.tenure);
  payload.MonthlyCharges = Number(payload.MonthlyCharges);
  payload.TotalCharges = Number(payload.TotalCharges);
  return payload;
}

function renderPrediction(result, requestId) {
  const probability = result.churn_probability;
  setText("probability", String(probability));
  setText("score-percent", `${(Number(probability) * 100).toFixed(1)}% score · exact value shown above`);
  scoreProgress.value = Number(probability);
  setText("decision", `${result.risk_level} (${result.churn_prediction})`);
  setText("threshold", `${result.decision_threshold} · ${humanizeStatus(result.threshold_status)}`);
  setText("calibration", humanizeStatus(result.calibration_status));
  setText("explanation", result.explanation_status === "not_available" ? "Not available" : result.explanation_status);
  setText("request-id", requestId || "Not supplied");
  decisionBadge.textContent = result.risk_level === "REVIEW" ? "Review indicated" : "Below threshold";
  decisionBadge.dataset.tone = result.risk_level === "REVIEW" ? "review" : "below";
  requiredElement("result-card").dataset.state = "complete";
}

function humanizeStatus(value) {
  const text = String(value).replaceAll("_", " ").replace("v1 1", "v1.1");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function validationSummary(detail) {
  if (!Array.isArray(detail) || detail.length === 0) return "Request validation failed.";
  return detail.map((item) => `${item.loc.join(".")}: ${item.msg}`).join("; ");
}

async function responseBody(response) {
  try { return await response.json(); }
  catch (_error) { return {}; }
}

async function requestPrediction(event) {
  event.preventDefault();
  errorOutput.textContent = "";
  submitButton.disabled = true;
  submitLabel.textContent = "Verifying request…";
  try {
    const response = await fetch("/predict", {method: "POST", headers: requestHeaders(true), body: JSON.stringify(payloadFromForm())});
    const body = await responseBody(response);
    if (!response.ok) {
      const requestId = response.headers.get("X-Request-ID");
      errorOutput.textContent = response.status === 422 ? validationSummary(body.detail) : `Prediction failed${requestId ? ` · request ${requestId}` : ""}.`;
      return;
    }
    renderPrediction(body, response.headers.get("X-Request-ID"));
  } catch (_error) {
    errorOutput.textContent = "Prediction service is unavailable. Verify the service and try again.";
    setServiceStatus("offline", "Service unavailable");
  } finally {
    submitButton.disabled = false;
    submitLabel.textContent = "Run verified prediction";
  }
}

function clearImportance() {
  while (importanceContainer.firstChild) importanceContainer.removeChild(importanceContainer.firstChild);
}

function readableFeatureName(name) {
  return name.replace(/^num__|^cat__/, "").replaceAll("_", " · ");
}

function renderGlobalImportance(items) {
  clearImportance();
  items.slice(0, 5).forEach((item) => {
    const row = document.createElement("div");
    row.className = "importance-row";
    const label = document.createElement("span");
    label.textContent = readableFeatureName(item.feature);
    const bar = document.createElement("progress");
    bar.max = 0.15;
    bar.value = Number(item.importance);
    bar.setAttribute("aria-label", `${label.textContent} global importance`);
    const value = document.createElement("b");
    value.textContent = Number(item.importance).toFixed(4);
    row.append(label, bar, value);
    importanceContainer.appendChild(row);
  });
}

async function loadModelInfo() {
  refreshModel.disabled = true;
  refreshModel.textContent = "Checking…";
  try {
    const response = await fetch("/model-info", {headers: requestHeaders()});
    if (!response.ok) {
      const status = response.status === 401 ? "API key required" : "Unavailable";
      setText("model-id", status);
      setText("final-test", status);
      return;
    }
    const manifest = await response.json();
    setText("model-id", manifest.model_id);
    setText("final-test", humanizeStatus(manifest.final_test.status));
    renderGlobalImportance(manifest.global_importance || []);
  } catch (_error) {
    setText("model-id", "Unavailable");
    setText("final-test", "Unavailable");
  } finally {
    refreshModel.disabled = false;
    refreshModel.textContent = "Connect";
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("health check failed");
    const health = await response.json();
    setServiceStatus(health.status === "healthy" ? "online" : "offline", health.status === "healthy" ? "Service healthy" : "Service degraded");
  } catch (_error) { setServiceStatus("offline", "Service unavailable"); }
}

function toggleApiKeyVisibility() {
  const revealing = apiKey.type === "password";
  apiKey.type = revealing ? "text" : "password";
  toggleKey.textContent = revealing ? "Hide" : "Show";
  toggleKey.setAttribute("aria-label", revealing ? "Hide API key" : "Show API key");
  toggleKey.setAttribute("aria-pressed", String(revealing));
}

function resetSample() {
  form.reset();
  errorOutput.textContent = "";
  syncInternetServices();
  syncPhoneServices();
  syncCharges();
}

internetService.addEventListener("change", syncInternetServices);
phoneService.addEventListener("change", syncPhoneServices);
tenure.addEventListener("input", syncCharges);
form.addEventListener("submit", requestPrediction);
toggleKey.addEventListener("click", toggleApiKeyVisibility);
refreshModel.addEventListener("click", loadModelInfo);
resetForm.addEventListener("click", resetSample);
apiKey.addEventListener("change", loadModelInfo);
syncInternetServices();
syncPhoneServices();
syncCharges();
loadHealth();
loadModelInfo();
