import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FineTuneDrawer } from "./FineTuneDrawer";
import { makeBootstrap } from "../test/fixtures";

describe("FineTuneDrawer", () => {
  it("keeps field and switch changes mapped to their settings keys", () => {
    const onChange = vi.fn();
    const settings = makeBootstrap({ use_llm: true }).settings;

    render(<FineTuneDrawer onChange={onChange} onClose={vi.fn()} open settings={settings} />);

    fireEvent.change(screen.getByLabelText("Language"), { target: { value: "en" } });
    fireEvent.change(screen.getByLabelText("Model endpoint"), { target: { value: "http://localhost:11434/v1" } });
    fireEvent.click(screen.getByLabelText("OCR scanned PDFs"));

    expect(onChange).toHaveBeenNthCalledWith(1, "language", "en");
    expect(onChange).toHaveBeenNthCalledWith(2, "llm_url", "http://localhost:11434/v1");
    expect(onChange).toHaveBeenNthCalledWith(3, "use_ocr", !settings.use_ocr);
  });

  it("retains the closed-drawer behavior", () => {
    render(<FineTuneDrawer onChange={vi.fn()} onClose={vi.fn()} open={false} settings={makeBootstrap().settings} />);

    expect(screen.queryByLabelText("Fine-tune settings")).not.toBeInTheDocument();
  });
});
