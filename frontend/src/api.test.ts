import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, errorMessage } from "./api";
import type { Settings } from "./types";

const settings = {
  directory: "/inbox",
  dry_run: true,
} as Settings;

const response = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

describe("API adapter", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses relative same-origin API paths and returns parsed JSON", async () => {
    const listing = { path: "/inbox", parent: null, entries: [], pdf_count: 0 };
    fetchMock.mockResolvedValue(response(listing));

    await expect(api.filesystem("/inbox/Quarterly report.pdf?draft=true")).resolves.toEqual(listing);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/filesystem?path=%2Finbox%2FQuarterly%20report.pdf%3Fdraft%3Dtrue",
      undefined,
    );
  });

  it("rejects paths that URL normalization moves outside the local API prefix", async () => {
    await expect(api.run("../../../outside-api")).rejects.toThrow(
      "Folionym only permits same-origin /api/v1 requests.",
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("checks the session before returning bootstrap JSON", async () => {
    const bootstrap = { settings, roots: [], capabilities: {} };
    fetchMock.mockResolvedValueOnce(response({ ready: true })).mockResolvedValueOnce(response(bootstrap));

    await expect(api.bootstrap()).resolves.toEqual(bootstrap);

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/v1/session", undefined);
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/v1/bootstrap", undefined);
  });

  it("sends the POST JSON contract for preview, cancellation, and apply actions", async () => {
    fetchMock
      .mockResolvedValueOnce(response({ run_id: "preview-1" }))
      .mockResolvedValueOnce(response({ id: "preview-1" }))
      .mockResolvedValueOnce(response({ run_id: "apply-1" }));

    await expect(api.startPreview("directory", "/inbox", settings, true)).resolves.toEqual({ run_id: "preview-1" });
    await expect(api.cancel("preview-1")).resolves.toEqual({ id: "preview-1" });
    await expect(api.apply("plan-1", 4, ["item-1", "item-2"])).resolves.toEqual({ run_id: "apply-1" });

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/v1/previews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_kind: "directory",
        path: "/inbox",
        settings,
        acknowledge_external_endpoint: true,
      }),
    });
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/v1/runs/preview-1/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    expect(fetchMock).toHaveBeenNthCalledWith(3, "/api/v1/plans/plan-1/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_revision: 4, selected_ids: ["item-1", "item-2"] }),
    });
  });

  it("keeps JSON error details on ApiError and exposes nested messages", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ detail: { code: "plan_conflict", message: "Plan changed" } }), {
        status: 409,
        statusText: "Conflict",
        headers: { "Content-Type": "application/json" },
      }),
    );

    const error = await api.plan("plan-1").catch((caught: unknown) => caught);

    expect(error).toMatchObject({
      name: "ApiError",
      status: 409,
      detail: { code: "plan_conflict", message: "Plan changed" },
    });
    expect(errorMessage(error)).toBe("Plan changed");
  });

  it("falls back to the HTTP status text when an error body is not JSON", async () => {
    fetchMock.mockResolvedValue(
      new Response("temporarily unavailable", {
        status: 503,
        statusText: "Service Unavailable",
      }),
    );

    const error = await api.report("report-1").catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 503, detail: "Service Unavailable" });
    expect(errorMessage(error)).toBe("Service Unavailable");
  });

  it("preserves network Errors and gives non-Errors the generic user message", async () => {
    fetchMock.mockRejectedValueOnce(new Error("Network unavailable")).mockRejectedValueOnce("offline");

    const networkError = await api.run("run-1").catch((caught: unknown) => caught);
    const nonError = await api.run("run-2").catch((caught: unknown) => caught);

    expect(errorMessage(networkError)).toBe("Network unavailable");
    expect(errorMessage(nonError)).toBe("Something went wrong.");
  });
});
