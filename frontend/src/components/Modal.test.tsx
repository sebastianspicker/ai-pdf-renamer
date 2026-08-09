import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal";

function modalContent() {
  return (
    <>
      <button>First control</button>
      <button>Last control</button>
    </>
  );
}

describe("Modal", () => {
  it("focuses its close button, wraps Tab keys, and closes on Escape", () => {
    const onClose = vi.fn();
    render(
      <Modal footer={<button>Footer control</button>} onClose={onClose} open title="Test modal">
        {modalContent()}
      </Modal>,
    );

    const closeButton = screen.getByRole("button", { name: "Close dialog" });
    const footerButton = screen.getByRole("button", { name: "Footer control" });
    expect(closeButton).toHaveFocus();

    footerButton.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(closeButton).toHaveFocus();

    closeButton.focus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(footerButton).toHaveFocus();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("removes key handling and restores the previously focused element when closed", () => {
    const onClose = vi.fn();
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const { rerender } = render(
      <Modal onClose={onClose} open title="Test modal">
        {modalContent()}
      </Modal>,
    );

    expect(screen.getByRole("button", { name: "Close dialog" })).toHaveFocus();
    rerender(
      <Modal onClose={onClose} open={false} title="Test modal">
        {modalContent()}
      </Modal>,
    );

    expect(trigger).toHaveFocus();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
    trigger.remove();
  });

  it("closes only for mousedown on the backdrop and retains optional rendering", () => {
    const onClose = vi.fn();
    render(
      <Modal description="Details" footer={<button>Footer control</button>} onClose={onClose} open title="Test modal" wide>
        {modalContent()}
      </Modal>,
    );

    const dialog = screen.getByRole("dialog", { name: "Test modal" });
    expect(screen.getByText("Details")).toBeInTheDocument();
    expect(screen.getByText("Footer control")).toBeInTheDocument();
    expect(dialog.querySelector(".modal")).toHaveClass("modal--wide");

    fireEvent.mouseDown(screen.getByRole("button", { name: "First control" }));
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.mouseDown(dialog);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
