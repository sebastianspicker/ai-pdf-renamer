import type {
  Bootstrap,
  DirectoryListing,
  Plan,
  Report,
  Run,
  Settings,
} from "./types";
import { ApiError } from "./api-errors";

export { ApiError, errorMessage } from "./api-errors";

const LOCAL_API_PREFIX = "/api/v1/";

function localApiPath(path: string): string {
  // Every frontend request remains beneath the API prefix on this page's origin.
  // Return only the relative path so the browser fetch cannot target another host.
  const url = new URL(path, window.location.origin);
  if (url.origin !== window.location.origin || !url.pathname.startsWith(LOCAL_API_PREFIX)) {
    throw new Error("Folionym only permits same-origin /api/v1 requests.");
  }
  return `${url.pathname}${url.search}`;
}

function detailFromPayload(payload: unknown): unknown {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    return payload.detail;
  }
  return payload;
}

function errorPayload(response: Response): Promise<unknown> {
  return response.json().then(detailFromPayload).catch(() => response.statusText);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await window.fetch(localApiPath(path), init);
  if (!response.ok) {
    throw new ApiError(response.status, await errorPayload(response));
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  bootstrap: async () => {
    await request<{ ready: boolean }>("/api/v1/session");
    return request<Bootstrap>("/api/v1/bootstrap");
  },
  filesystem: (path: string) =>
    request<DirectoryListing>(`/api/v1/filesystem?path=${encodeURIComponent(path)}`),
  startPreview: (
    sourceKind: "directory" | "file",
    path: string,
    settings: Settings,
    acknowledge = false,
  ) =>
    request<{ run_id: string }>(
      "/api/v1/previews",
      json({
        source_kind: sourceKind,
        path,
        settings,
        acknowledge_external_endpoint: acknowledge,
      }),
    ),
  run: (id: string) => request<Run>(`/api/v1/runs/${id}`),
  cancel: (id: string) => request<Run>(`/api/v1/runs/${id}/cancel`, json({})),
  plan: (id: string) => request<Plan>(`/api/v1/plans/${id}`),
  apply: (id: string, revision: number, selectedIds: string[]) =>
    request<{ run_id: string }>(
      `/api/v1/plans/${id}/apply`,
      json({ plan_revision: revision, selected_ids: selectedIds }),
    ),
  report: (id: string) => request<Report>(`/api/v1/reports/${id}`),
};
