import { type MouseEvent, type ReactNode, type RefObject, useEffect, useId, useRef } from "react";
import { CloseIcon } from "../icons";

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), input:not([disabled]), select:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

function keepFocusInModal(event: KeyboardEvent, modal: HTMLElement | null) {
  if (event.key !== "Tab" || !modal) return;
  const focusable = [...modal.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)];
  const first = focusable[0];
  const last = focusable.at(-1);
  if (!first || !last) return;

  const wrapFrom = event.shiftKey ? first : last;
  if (document.activeElement !== wrapFrom) return;
  event.preventDefault();
  (event.shiftKey ? last : first).focus();
}

function handleModalKeyDown(event: KeyboardEvent, modal: HTMLElement | null, onClose: () => void) {
  if (event.key === "Escape") {
    onClose();
    return;
  }
  keepFocusInModal(event, modal);
}

function useModalFocus(
  open: boolean,
  onClose: () => void,
  closeRef: RefObject<HTMLButtonElement | null>,
  modalRef: RefObject<HTMLElement | null>,
) {
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => handleModalKeyDown(event, modalRef.current, onClose);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previous?.focus();
    };
  }, [open, onClose, closeRef, modalRef]);
}

function closeFromBackdrop(event: MouseEvent<HTMLDivElement>, onClose: () => void) {
  if (event.target === event.currentTarget) onClose();
}

export function Modal({
  open,
  title,
  description,
  children,
  footer,
  onClose,
  wide = false,
}: {
  open: boolean;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const modalRef = useRef<HTMLElement>(null);
  const titleId = useId();
  useModalFocus(open, onClose, closeRef, modalRef);
  if (!open) return null;
  return (
    <div
      aria-labelledby={titleId}
      aria-modal="true"
      className="modal-layer"
      onMouseDown={(event) => closeFromBackdrop(event, onClose)}
      role="dialog"
    >
      <section className={`modal ${wide ? "modal--wide" : ""}`} ref={modalRef}>
        <header className="modal__header">
          <div>
            <h2 id={titleId}>{title}</h2>
            {description && <p>{description}</p>}
          </div>
          <button aria-label="Close dialog" className="icon-button" onClick={onClose} ref={closeRef}>
            <CloseIcon />
          </button>
        </header>
        <div className="modal__body">{children}</div>
        {footer && <footer className="modal__footer">{footer}</footer>}
      </section>
    </div>
  );
}
