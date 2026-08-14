import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const baseUrl = process.env.FOLIONYM_SCREENSHOT_URL || "http://127.0.0.1:8765";
const sourcePath = process.env.FOLIONYM_SCREENSHOT_SOURCE;
const sourceCount = process.env.FOLIONYM_SCREENSHOT_COUNT || "8";
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const screenshotRoot = path.resolve(scriptDir, "../../docs/screenshots");

function screenshotOutputDirectory(output = process.env.FOLIONYM_SCREENSHOT_OUTPUT) {
  const candidate = path.resolve(output || screenshotRoot);
  const relativePath = path.relative(screenshotRoot, candidate);
  if (relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
    throw new Error("FOLIONYM_SCREENSHOT_OUTPUT must stay inside docs/screenshots.");
  }
  return candidate;
}

const outputDir = screenshotOutputDirectory();

if (!sourcePath) {
  throw new Error("FOLIONYM_SCREENSHOT_SOURCE must name the synthetic PDF folder.");
}

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  colorScheme: "light",
  locale: "en-GB",
  reducedMotion: "reduce",
  viewport: { width: 1440, height: 1000 },
});
const page = await context.newPage();
const consoleProblems = [];

async function captureScreenshot(name, fullPage) {
  await page.screenshot({
    path: path.join(outputDir, name),
    fullPage,
  });
}

page.on("console", (message) => {
  if (message.type() === "error" || message.type() === "warning") {
    consoleProblems.push(`${message.type()}: ${message.text()}`);
  }
});
page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));
page.on("response", (response) => {
  if (response.url().includes("/api/") && !response.ok()) {
    consoleProblems.push(`http ${response.status()}: ${response.url()}`);
  }
});

await page.goto(baseUrl, { waitUntil: "networkidle" });
await page.getByRole("heading", { name: "Choose local PDFs" }).waitFor();
await page.getByRole("button", { name: "Browse" }).click();
await page.getByRole("textbox", { name: "Folder path" }).fill(sourcePath);
await page.getByRole("button", { name: "Go", exact: true }).click();
await page.getByText(`${sourceCount} PDFs in this folder`, { exact: true }).waitFor();
await page.getByRole("button", { name: "Choose folder" }).click();
// Scope may appear as full path or compact ellipsis form in the instrument bar.
// The path is passed to Playwright as text, never interpolated into page HTML or a selector.
await page
  .locator(".source-picker__copy strong, .scope-path")
  .filter({ hasText: sourcePath.split("/").filter(Boolean).at(-1) || sourcePath })
  .first()
  .waitFor();

// Deterministic, local-only capture: heuristics without an HTTP model endpoint.
await page.getByRole("button", { name: /Fine-tune/ }).click();
const modelToggle = page.getByRole("checkbox", { name: /Model assistance/i });
if (await modelToggle.isChecked()) {
  await modelToggle.click();
}
await page.getByRole("button", { name: "Done" }).click();

await captureScreenshot("web-source-desktop.png", true);

await page.getByRole("button", { name: /Build preview/ }).click();
await page.waitForURL("**/preview", { timeout: 60_000 });
await page.getByRole("heading", { name: "Rename ledger" }).waitFor();
await page.getByRole("listbox", { name: "Proposed renames" }).waitFor();
await page.waitForFunction(() => {
  const image =
    document.querySelector(".thumb img") || document.querySelector(".document-preview img");
  return image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0;
});
await captureScreenshot("web-preview-desktop.png", true);

await page.setViewportSize({ width: 820, height: 1180 });
await captureScreenshot("web-preview-tablet.png", false);

await page.setViewportSize({ width: 390, height: 844 });
await captureScreenshot("web-preview-mobile.png", false);

await page.getByRole("button", { name: /Apply \d+ names?/ }).click();
await page.getByRole("heading", { name: /Write \d+ selected names?/ }).waitFor();
await captureScreenshot("web-apply-confirm-mobile.png", false);

await page.getByRole("button", { name: "Rename files" }).click();
await page.waitForURL("**/apply", { timeout: 60_000 });
await page.getByText("Apply", { exact: true }).first().waitFor();
await page.getByRole("heading", { name: /file(s)? renamed|Run finished with issues/ }).waitFor();
await captureScreenshot("web-apply-mobile.png", false);

await page.setViewportSize({ width: 1440, height: 1000 });
await captureScreenshot("web-apply-desktop.png", true);

if (consoleProblems.length) {
  throw new Error(`Browser console problems:\n${consoleProblems.join("\n")}`);
}

console.log(`Captured Folionym browser screenshots in ${outputDir}`);
await browser.close();
