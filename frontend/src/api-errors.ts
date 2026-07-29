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
