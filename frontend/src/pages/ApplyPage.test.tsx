import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";
import { makeBootstrap } from "../test/fixtures";
import type { Report } from "../types";
import { ApplyPage } from "./ApplyPage";

const { navigateMock } = vi.hoisted(() => ({ navigateMock: vi.fn() }));

vi.mock("../lib/routing", () => ({ navigate: navigateMock }));

const report: Report = {
  id: "report-456",
  plan_id: "plan-123",
  source: "/tmp/pdfs",
  started_at: "2026-08-06T10:00:00Z",
  completed_at: "2026-08-06T10:01:00Z",
  counts: { renamed: 1, skipped: 1, unchanged: 1, failed: 1, cancelled: 1 },
  items: [
    { item_id: "renamed-1", source_name: "receipt.pdf", target_name: "20260806-receipt.pdf", status: "renamed", reason: null },
    { item_id: "skipped-1", source_name: "already-named.pdf", target_name: null, status: "skipped", reason: "Already named" },
    { item_id: "unchanged-1", source_name: "same.pdf", target_name: "same.pdf", status: "unchanged", reason: null },
    { item_id: "failed-1", source_name: "blocked.pdf", target_name: "blocked.pdf", status: "failed", reason: "Collision" },
    { item_id: "cancelled-1", source_name: "later.pdf", target_name: "later.pdf", status: "cancelled", reason: "Stopped" },
  ],
};

function renderApply(bootstrap = makeBootstrap()) {
  sessionStorage.setItem("folionym.report", report.id);
  render(<ApplyPage bootstrap={bootstrap} />);
}

beforeEach(() => {
  sessionStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  navigateMock.mockReset();
});

describe("ApplyPage", () => {
  it("loads the report and shows the partial outcome summary", async () => {
    const getReport = vi.spyOn(api, "report").mockResolvedValue(report);
    renderApply();

    expect(screen.getByText("Loading report")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Run finished with issues" })).toBeInTheDocument();
    expect(getReport).toHaveBeenCalledWith(report.id);
    expect(screen.getByLabelText("Apply totals")).toHaveTextContent("1Renamed1Unchanged1Failed1Cancelled");
    expect(screen.getByText("Validated renames were written. Items that failed checks were left unchanged.")).toBeInTheDocument();
  });

  it("returns to source without requesting a report when the session is missing", async () => {
    const getReport = vi.spyOn(api, "report");
    render(<ApplyPage bootstrap={makeBootstrap()} />);

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith("source");
    });
    expect(getReport).not.toHaveBeenCalled();
  });

  it("filters exact statuses while keeping skipped items in all and displaying them as unchanged", async () => {
    vi.spyOn(api, "report").mockResolvedValue(report);
    renderApply();

    await screen.findByRole("heading", { name: "Run finished with issues" });
    expect(screen.getByText("already-named.pdf")).toBeInTheDocument();
    const ledger = screen.getByText("already-named.pdf").closest(".result-table") as HTMLDivElement;
    expect(ledger).toBeInTheDocument();
    expect(within(ledger).getAllByText("Unchanged")).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "unchanged" }));
    expect(screen.getAllByText("same.pdf")).toHaveLength(2);
    expect(screen.queryByText("already-named.pdf")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "failed" }));
    expect(screen.getAllByText("blocked.pdf")).toHaveLength(2);
    expect(screen.queryByText("same.pdf")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "skipped" })).not.toBeInTheDocument();
  });

  it("shows API errors and provides a route back to source", async () => {
    vi.spyOn(api, "report").mockRejectedValue(new Error("Report expired"));
    renderApply();

    expect(await screen.findByRole("heading", { name: "Report unavailable" })).toBeInTheDocument();
    expect(screen.getByText("Report expired")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Back to Source" }));
    expect(navigateMock).toHaveBeenCalledWith("source");
  });
});
