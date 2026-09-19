"use strict";

const endpointContainer = document.getElementById("api-endpoints");
const schemaOutput = document.getElementById("schema-json");

function clear(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderEndpoints(paths) {
  clear(endpointContainer);
  Object.entries(paths).forEach(([path, operations]) => {
    Object.entries(operations).forEach(([method, operation]) => {
      const row = element("article", "api-row");
      const methodLabel = element("span", `http-method ${method.toLowerCase()}`, method.toUpperCase());
      const copy = element("div", "api-row-copy");
      copy.append(element("strong", "", path), element("p", "", operation.summary || "API operation"));
      const auth = element("span", "badge", ["/health", "/", "/docs", "/input-schema"].includes(path) ? "Public" : "API key");
      row.append(methodLabel, copy, auth);
      endpointContainer.appendChild(row);
    });
  });
}

async function loadContract() {
  try {
    const response = await fetch("/openapi.json");
    if (!response.ok) throw new Error("OpenAPI unavailable");
    const contract = await response.json();
    renderEndpoints(contract.paths || {});
    const schema = contract.components?.schemas?.CustomerPayload || {};
    schemaOutput.textContent = JSON.stringify(schema, null, 2);
  } catch (_error) {
    clear(endpointContainer);
    endpointContainer.appendChild(element("p", "docs-error", "The OpenAPI contract is currently unavailable."));
    schemaOutput.textContent = "Schema unavailable.";
  }
}

loadContract();
