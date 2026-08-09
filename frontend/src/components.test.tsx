import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { StatusPill, StageSpine, compactPath, formatBytes } from "./components";
import { SourcePage } from "./pages/SourcePage";
import { makeBootstrap } from "./test/fixtures";

const { useRunMock } = vi.hoisted(() => ({ useRunMock: vi.fn() }));

vi.mock("./hooks/useRun", () => ({ useRun: useRunMock }));

function renderSourcePage() {
  const bootstrap = makeBootstrap({
    llm_url: "https://fallback.example.test/v1",
    use_llm: true,
  });
  const onBootstrapChange = vi.fn();
  render(<SourcePage bootstrap={bootstrap} onBootstrapChange={onBootstrapChange} />);
  return { bootstrap, onBootstrapChange };
}

beforeEach(() => {
  useRunMock.mockReturnValue({ error: "", run: null, setError: vi.fn() });
});

afterEach(() => {
  vi.restoreAllMocks();
  useRunMock.mockReset();
});

describe("shared frontend primitives", () => {
  it("marks the active workflow stage and completed predecessors", () => {
    render(<StageSpine active={2} />);

    expect(screen.getByText("Preview").closest(".stage")).toHaveAttribute("aria-current", "step");
    expect(screen.getByText("Source").closest(".stage")).toHaveClass("stage--done");
    expect(screen.getByText("Preview").closest(".stage")).toHaveClass("stage--current");
  });

  it("renders factual status language in sentence case", () => {
    render(<StatusPill status="review" />);
    expect(screen.getByText("Review")).toHaveClass("status--review");
  });

  it("formats paths and file sizes for dense data views", () => {
    expect(compactPath("/Users/ada/Documents/Inbox")).toBe("…/Documents/Inbox");
    expect(formatBytes(2048)).toBe("2 KB");
  });

  it("keeps empty-source validation local", () => {
    const startPreview = vi.spyOn(api, "startPreview");
    renderSourcePage();

    fireEvent.click(screen.getByRole("button", { name: "Single PDF" }));
    fireEvent.click(screen.getByRole("button", { name: /Build preview/ }));

    expect(screen.getByText("Choose a PDF before continuing.")).toBeInTheDocument();
    expect(startPreview).not.toHaveBeenCalled();
  });

  it("acknowledges only the required external endpoint response and retries successfully", async () => {
    useRunMock.mockReturnValue({ error: "", run: null, setError: vi.fn() });
    const startPreview = vi
      .spyOn(api, "startPreview")
      .mockRejectedValueOnce(
        new ApiError(409, {
          code: "external_endpoint_ack_required",
          endpoint: null,
        }),
      )
      .mockResolvedValueOnce({ run_id: "run-123" });
    const { bootstrap, onBootstrapChange } = renderSourcePage();

    fireEvent.click(screen.getByRole("button", { name: /Build preview/ }));

    expect(await screen.findByRole("dialog")).toHaveTextContent("https://fallback.example.test/v1");
    fireEvent.click(screen.getByRole("button", { name: "Continue once" }));

    await waitFor(() => {
      expect(onBootstrapChange).toHaveBeenCalledWith({ ...bootstrap, settings: bootstrap.settings });
      expect(useRunMock).toHaveBeenLastCalledWith("run-123", expect.any(Function));
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(startPreview).toHaveBeenNthCalledWith(1, "directory", "/tmp/pdfs", bootstrap.settings, false);
    expect(startPreview).toHaveBeenNthCalledWith(2, "directory", "/tmp/pdfs", bootstrap.settings, true);
  });

  it.each([
    [new ApiError(409, { code: "external_endpoint_ack_required" }), "The request could not be completed."],
    [new ApiError(409, { code: "other", endpoint: "https://example.test", message: "Unexpected response" }), "Unexpected response"],
    [new ApiError(500, { code: "external_endpoint_ack_required", endpoint: "https://example.test", message: "Server error" }), "Server error"],
    [new Error("Network error"), "Network error"],
  ])("shows ordinary errors for non-acknowledgement responses", async (requestError, message) => {
    vi.spyOn(api, "startPreview").mockRejectedValueOnce(requestError);
    renderSourcePage();

    fireEvent.click(screen.getByRole("button", { name: /Build preview/ }));

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
