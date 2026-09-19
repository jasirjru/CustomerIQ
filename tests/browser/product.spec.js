import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("renders the exact API score without browser-side adjustment", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Service healthy")).toBeVisible();
  const responsePromise = page.waitForResponse((response) => response.url().endsWith("/predict"));
  await page.getByRole("button", { name: "Run verified prediction" }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const apiResult = await response.json();
  await expect(page.getByLabel("Exact churn probability")).toHaveText(String(apiResult.churn_probability));
  await expect(page.getByText("Not fitted", { exact: true })).toBeVisible();
  await expect(page.getByText("Not available", { exact: true })).toBeVisible();
});

test("synchronizes telecom cross-field states before submission", async ({ page }) => {
  await page.goto("/#studio");
  await page.locator('select[name="PhoneService"]').selectOption("No");
  await expect(page.locator('select[name="MultipleLines"]')).toHaveValue("No phone service");
  await page.locator('select[name="InternetService"]').selectOption("No");
  for (const field of await page.locator("[data-internet-addon]").all()) {
    await expect(field).toHaveValue("No internet service");
  }
  await page.locator('input[name="tenure"]').fill("0");
  await expect(page.locator('input[name="TotalCharges"]')).toHaveValue("0");
  await expect(page.locator('input[name="TotalCharges"]')).toHaveAttribute("readonly", "");
});

test("does not execute customer-controlled markup", async ({ page }) => {
  await page.addInitScript(() => { window.__customerIqXss = false; });
  await page.goto("/#studio");
  await page.locator('input[name="customer_id"]').fill('<img src=x onerror="window.__customerIqXss=true">');
  await page.getByRole("button", { name: "Run verified prediction" }).click();
  await expect(page.getByLabel("Exact churn probability")).not.toHaveText("—");
  expect(await page.evaluate(() => window.__customerIqXss)).toBe(false);
});

test("has no automatically detectable WCAG A/AA violations", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
    .analyze();
  expect(results.violations).toEqual([]);
});

test("local API reference is generated and accessible", async ({ page }) => {
  await page.goto("/docs");
  await expect(page.getByRole("heading", { name: "Endpoints" })).toBeVisible();
  await expect(page.locator(".api-row")).toHaveCount(5);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
});

test("layout does not overflow the viewport", async ({ page }) => {
  await page.goto("/");
  const sizes = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: document.documentElement.clientWidth }));
  expect(sizes.width).toBeLessThanOrEqual(sizes.viewport);
});

test("loads the product without third-party or failed asset requests", async ({ page }) => {
  const failures = [];
  page.on("requestfailed", (request) => failures.push(request.url()));
  const responses = [];
  page.on("response", (response) => responses.push({ url: response.url(), status: response.status() }));
  await page.goto("/");
  await expect(page.getByText("Service healthy")).toBeVisible();
  expect(failures).toEqual([]);
  expect(responses.every(({ url }) => new URL(url).origin === "http://127.0.0.1:8765")).toBe(true);
  expect(responses.filter(({ status }) => status >= 400)).toEqual([]);
});
