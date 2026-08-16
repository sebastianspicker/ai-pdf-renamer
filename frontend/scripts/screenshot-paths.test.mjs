import { mkdtemp, mkdir, realpath, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, expect, test } from "vitest";

import {
  resolveScreenshotFile,
  resolveScreenshotOutputDirectory,
} from "./screenshot-paths.mjs";

const temporaryRoots = [];

async function temporaryRoot() {
  const root = await mkdtemp(path.join(os.tmpdir(), "folionym-screenshots-"));
  temporaryRoots.push(root);
  return root;
}

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

test("accepts an existing real output directory inside the screenshot root", async () => {
  const root = await temporaryRoot();
  const screenshotRoot = path.join(root, "screenshots");
  const output = path.join(screenshotRoot, "run");
  await mkdir(output, { recursive: true });

  expect(await resolveScreenshotOutputDirectory(screenshotRoot, output)).toBe(await realpath(output));
});

test("rejects lexical escapes and symlinked output directories", async () => {
  const root = await temporaryRoot();
  const screenshotRoot = path.join(root, "screenshots");
  const outside = path.join(root, "outside");
  await mkdir(screenshotRoot);
  await mkdir(outside);
  await symlink(outside, path.join(screenshotRoot, "linked"));

  await expect(resolveScreenshotOutputDirectory(screenshotRoot, outside)).rejects.toThrow(
    /must stay inside/u,
  );
  await expect(
    resolveScreenshotOutputDirectory(screenshotRoot, path.join(screenshotRoot, "linked")),
  ).rejects.toThrow(/real directory, not a symlink/u);
});

test("rejects unsafe names and existing symlink targets", async () => {
  const root = await temporaryRoot();
  const target = path.join(root, "target.png");
  await writeFile(target, "not an image");
  await symlink(target, path.join(root, "capture.png"));

  await expect(resolveScreenshotFile(root, "../escape.png")).rejects.toThrow(/simple lowercase/u);
  await expect(resolveScreenshotFile(root, "capture.png")).rejects.toThrow(/regular files/u);
});
