import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { makeBootstrap } from "../test/fixtures";
import type { Plan, Run } from "../types";
import { PreviewPage } from "./PreviewPage";

const { navigateMock, useRunMock } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  useRunMock: vi.fn(),
}));

vi.mock("../hooks/useRun", () => ({ useRun: useRunMock }));
vi.mock("../lib/routing", () => ({ navigate: navigateMock }));

const plan: Plan = {
  id: "plan-123",
  revision: 4,
  source: "/tmp/pdfs",
  source_kind: "directory",
  created_at: "2026-08-06T10:00:00Z",
  counts: { all: 3, ready: 1, review: 1, skipped: 1, failed: 0 },
  items: [
    {
      id: "ready-1",
      current_name: "receipt.pdf",
      source_path: "/tmp/pdfs/receipt.pdf",
      proposed_name: "20260806-receipt.pdf",
      status: "ready",
      included: true,
      reason: null,
      size: 32,
      modified_at: null,
      metadata: {},
    },
    {
      id: "review-1",
      current_name: "letter.pdf",
      source_path: "/tmp/pdfs/letter.pdf",
      proposed_name: "20260806-letter.pdf",
      status: "review",
      included: false,
      reason: "Check date",
      size: 64,
      modified_at: null,
      metadata: {},
    },
    {
      id: "skipped-1",
      current_name: "already-named.pdf",
      source_path: "/tmp/pdfs/already-named.pdf",
      proposed_name: null,
      status: "skipped",
      included: true,
      reason: "Already named",
      size: 16,
      modified_at: null,
      metadata: {},
    },
  ],
};

function renderPreview(bootstrap = makeBootstrap()) {
  sessionStorage.setItem("folionym.plan", plan.id);
  render(<PreviewPage bootstrap={bootstrap} />);
}

beforeEach(() => {
  sessionStorage.clear();
  useRunMock.mockReturnValue({ error: "", run: null, setError: vi.fn() });
});

afterEach(() => {
  vi.restoreAllMocks();
  navigateMock.mockReset();
  useRunMock.mockReset();
});

describe("PreviewPage", () => {
  it("returns to source when the preview session is missing", async () => {
    const getPlan = vi.spyOn(api, "plan");
    render(<PreviewPage bootstrap={makeBootstrap()} />);

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith("source");
    });
    expect(getPlan).not.toHaveBeenCalled();
  });

  it("shows the request error when the preview plan cannot be loaded", async () => {
    vi.spyOn(api, "plan").mockRejectedValue(new Error("Plan expired"));
    renderPreview();

    expect(await screen.findByRole("heading", { name: "Preview unavailable" })).toBeInTheDocument();
    expect(screen.getByText("Plan expired")).toBeInTheDocument();
  });

  it("filters current and proposed filenames without case sensitivity", async () => {
    vi.spyOn(api, "plan").mockResolvedValue(plan);
    renderPreview();

    const ledger = await screen.findByRole("listbox", { name: "Proposed renames" });
    fireEvent.change(screen.getByRole("searchbox", { name: "Search filenames" }), {
      target: { value: "LETTER" },
    });
    expect(within(ledger).getByText("letter.pdf")).toBeInTheDocument();
    expect(within(ledger).queryByText("receipt.pdf")).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole("searchbox", { name: "Search filenames" }), {
      target: { value: "20260806-RECEIPT" },
    });
    expect(within(ledger).getByText("receipt.pdf")).toBeInTheDocument();
    expect(within(ledger).queryByText("letter.pdf")).not.toBeInTheDocument();
  });

  it("initializes included selections and selects only visible ready or review items", async () => {
    vi.spyOn(api, "plan").mockResolvedValue(plan);
    renderPreview();

    await screen.findByRole("listbox", { name: "Proposed renames" });
    expect(screen.getByRole("checkbox", { name: "Include 20260806-receipt.pdf" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Include 20260806-letter.pdf" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Include already-named.pdf" })).toBeChecked();
    expect(screen.getByAltText("First page of receipt.pdf")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Review 1/ }));
    fireEvent.click(screen.getByRole("button", { name: "Select visible" }));

    expect(screen.getByRole("button", { name: "Apply 3 names" })).toBeEnabled();
    fireEvent.click(screen.getByRole("tab", { name: /Skipped 1/ }));
    expect(screen.getByRole("button", { name: "Select visible" })).toBeDisabled();
  });

  it("applies the exact selected item ids with the plan revision", async () => {
    vi.spyOn(api, "plan").mockResolvedValue(plan);
    const apply = vi.spyOn(api, "apply").mockResolvedValue({ run_id: "run-123" });
    renderPreview();

    await screen.findByRole("listbox", { name: "Proposed renames" });
    fireEvent.click(screen.getByRole("button", { name: "Apply 2 names" }));
    fireEvent.click(screen.getByRole("button", { name: "Rename files" }));

    await waitFor(() => {
      expect(apply).toHaveBeenCalledWith(plan.id, plan.revision, ["ready-1", "skipped-1"]);
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("stores the completed report and navigates to apply", async () => {
    let onComplete: ((run: Run) => void) | undefined;
    vi.spyOn(api, "plan").mockResolvedValue(plan);
    vi.spyOn(api, "apply").mockResolvedValue({ run_id: "run-123" });
    useRunMock.mockImplementation((runId, callback) => {
      if (runId === "run-123") onComplete = callback;
      return { error: "", run: null, setError: vi.fn() };
    });
    renderPreview();

    await screen.findByRole("listbox", { name: "Proposed renames" });
    fireEvent.click(screen.getByRole("button", { name: "Apply 2 names" }));
    fireEvent.click(screen.getByRole("button", { name: "Rename files" }));

    await waitFor(() => {
      expect(onComplete).toBeTypeOf("function");
    });
    act(() => {
      onComplete?.({
        id: "run-123",
        kind: "apply",
        state: "completed",
        completed: 2,
        total: 2,
        current_file: "",
        message: "Done",
        plan_id: plan.id,
        report_id: "report-456",
        error: null,
      });
    });

    expect(sessionStorage.getItem("folionym.report")).toBe("report-456");
    expect(navigateMock).toHaveBeenCalledWith("apply");
  });
});
