import type {
  Bootstrap,
  DirectoryListing,
  Plan,
  Report,
  Run,
  Settings,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : "The request could not be completed.");
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, payload.detail ?? payload);
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

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail;
    if (typeof detail === "object" && detail !== null && "message" in detail) {
      return String(detail.message);
    }
    return error.message;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}
