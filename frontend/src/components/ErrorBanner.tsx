import { CloseIcon } from "../icons";

export function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss?: () => void;
}) {
  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      {onDismiss && (
        <button aria-label="Dismiss error" className="icon-button" onClick={onDismiss}>
          <CloseIcon size={16} />
        </button>
      )}
    </div>
  );
}
