import { lstat, realpath } from "node:fs/promises";
import path from "node:path";

function isContained(root, candidate) {
  const relativePath = path.relative(root, candidate);
  return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

async function requireDirectory(candidate, label) {
  let metadata;
  try {
    metadata = await lstat(candidate);
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new Error(`${label} must already exist.`, { cause: error });
    }
    throw error;
  }
  if (metadata.isSymbolicLink() || !metadata.isDirectory()) {
    throw new Error(`${label} must be a real directory, not a symlink.`);
  }
}

export async function resolveScreenshotOutputDirectory(screenshotRoot, requestedOutput) {
  await requireDirectory(screenshotRoot, "The screenshot root");
  const canonicalRoot = await realpath(screenshotRoot);
  const requestedPath = path.resolve(requestedOutput || screenshotRoot);
  if (!isContained(path.resolve(screenshotRoot), requestedPath)) {
    throw new Error("FOLIONYM_SCREENSHOT_OUTPUT must stay inside docs/screenshots.");
  }

  await requireDirectory(requestedPath, "FOLIONYM_SCREENSHOT_OUTPUT");
  const canonicalOutput = await realpath(requestedPath);
  if (!isContained(canonicalRoot, canonicalOutput)) {
    throw new Error("FOLIONYM_SCREENSHOT_OUTPUT must not escape through a symlink.");
  }
  return canonicalOutput;
}

export async function resolveScreenshotFile(outputDirectory, name) {
  if (path.basename(name) !== name || !/^[a-z0-9-]+\.png$/u.test(name)) {
    throw new Error("Screenshot names must be simple lowercase PNG filenames.");
  }

  const candidate = path.join(outputDirectory, name);
  try {
    const metadata = await lstat(candidate);
    if (metadata.isSymbolicLink() || !metadata.isFile()) {
      throw new Error("Existing screenshot targets must be regular files, not symlinks.");
    }
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }
  return candidate;
}
