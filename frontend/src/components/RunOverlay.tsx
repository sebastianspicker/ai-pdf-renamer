import { terminalStates } from "../hooks/useRun";
import type { Run } from "../types";
import { Button } from "./Button";
import { ErrorBanner } from "./ErrorBanner";
import { Modal } from "./Modal";

export function RunOverlay({
  run,
  error,
  onCancel,
  onClose,
}: {
  run: Run | null;
  error: string;
  onCancel: () => void;
  onClose: () => void;
}) {
  const isTerminal = run ? terminalStates.has(run.state) : false;
  const percentage = run?.total ? Math.round((run.completed / run.total) * 100) : 0;
  return (
    <Modal
      description={
        run?.kind === "apply"
          ? "Writing only the exact names you approved."
          : "Reading local PDFs and building filename proposals."
      }
      onClose={isTerminal || error ? onClose : () => undefined}
      open={Boolean(run) || Boolean(error)}
      title={run?.kind === "apply" ? "Applying names" : "Building preview"}
    >
      {error ? (
        <ErrorBanner message={error} />
      ) : (
        <div className="run-status">
          <div className="run-status__row">
            <strong>{run?.message}</strong>
            <span>{run?.total ? `${run.completed} / ${run.total}` : "Starting…"}</span>
          </div>
          <div
            aria-label={`${percentage}% complete`}
            aria-valuemax={100}
            aria-valuemin={0}
            aria-valuenow={percentage}
            className="progress"
            role="progressbar"
          >
            <span style={{ width: `${percentage}%` }} />
          </div>
          <p>{run?.current_file ?? "Preparing…"}</p>
          {run?.error && <ErrorBanner message={run.error} />}
          {!isTerminal && (
            <Button onClick={onCancel} variant="secondary">
              Cancel after current file
            </Button>
          )}
        </div>
      )}
    </Modal>
  );
}
