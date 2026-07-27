import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusPill, StageSpine, compactPath, formatBytes } from "./components";

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
});
