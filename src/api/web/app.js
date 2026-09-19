"use strict";

const form = document.getElementById("prediction-form");
const submitButton = document.getElementById("submit");
const errorOutput = document.getElementById("error");
const internetService = form.elements.InternetService;
const phoneService = form.elements.PhoneService;
const multipleLines = form.elements.MultipleLines;
const tenure = form.elements.tenure;
const totalCharges = form.elements.TotalCharges;
const apiKey = document.getElementById("api-key");

function requestHeaders() {
  const headers = {"Content-Type": "application/json"};
  if (apiKey.value) headers["X-API-Key"] = apiKey.value;
  return headers;
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function syncInternetServices() {
  const absent = internetService.value === "No";
  document.querySelectorAll("[data-internet-addon]").forEach((field) => {
    if (absent) {
      field.value = "No internet service";
    } else if (field.value === "No internet service") {
      field.value = "No";
    }
  });
}

function syncPhoneServices() {
  if (phoneService.value === "No") {
    multipleLines.value = "No phone service";
  } else if (multipleLines.value === "No phone service") {
    multipleLines.value = "No";
  }
}

function syncCharges() {
  const newAccount = Number(tenure.value) === 0;
  if (newAccount) {
    totalCharges.value = "0";
  }
  totalCharges.readOnly = newAccount;
}

function payloadFromForm() {
  const values = new FormData(form);
  const payload = Object.fromEntries(values.entries());
  if (!payload.customer_id.trim()) {
    delete payload.customer_id;
  }
  payload.SeniorCitizen = Number(payload.SeniorCitizen);
  payload.tenure = Number(payload.tenure);
  payload.MonthlyCharges = Number(payload.MonthlyCharges);
  payload.TotalCharges = Number(payload.TotalCharges);
  return payload;
}

function renderPrediction(result) {
  const probability = result.churn_probability;
  setText("probability", String(probability));
  setText("decision", `${result.risk_level} (${result.churn_prediction})`);
  setText("threshold", `${result.decision_threshold} — ${result.threshold_status}`);
  setText("calibration", result.calibration_status);
  setText("explanation", result.explanation_status === "not_available" ? "Not available" : result.explanation_status);
}

function validationSummary(detail) {
  if (!Array.isArray(detail) || detail.length === 0) {
    return "Request validation failed.";
  }
  return detail.map((item) => `${item.loc.join(".")}: ${item.msg}`).join("; ");
}

async function requestPrediction(event) {
  event.preventDefault();
  errorOutput.textContent = "";
  submitButton.disabled = true;
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: requestHeaders(),
      body: JSON.stringify(payloadFromForm()),
    });
    const body = await response.json();
    if (!response.ok) {
      errorOutput.textContent = response.status === 422
        ? validationSummary(body.detail)
        : "Prediction failed. Use the response request ID when contacting support.";
      return;
    }
    renderPrediction(body);
  } catch (_error) {
    errorOutput.textContent = "Prediction service is unavailable.";
  } finally {
    submitButton.disabled = false;
  }
}

async function loadModelInfo() {
  try {
    const response = await fetch("/model-info", {headers: requestHeaders()});
    if (!response.ok) return;
    const manifest = await response.json();
    setText("model-id", manifest.model_id);
    setText("final-test", manifest.final_test.status);
  } catch (_error) {
    setText("model-id", "Unavailable");
    setText("final-test", "Unavailable");
  }
}

internetService.addEventListener("change", syncInternetServices);
phoneService.addEventListener("change", syncPhoneServices);
tenure.addEventListener("change", syncCharges);
form.addEventListener("submit", requestPrediction);
apiKey.addEventListener("change", loadModelInfo);
syncInternetServices();
syncPhoneServices();
syncCharges();
loadModelInfo();
