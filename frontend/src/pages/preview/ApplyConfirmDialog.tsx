import { Button, Modal, compactPath } from "../../components";

type ApplyConfirmDialogProps = {
  open: boolean;
  selectedCount: number;
  reviewCount: number;
  skippedFailedCount: number;
  source: string;
  localOnly: boolean;
  onClose: () => void;
  onConfirm: () => void;
};

export function ApplyConfirmDialog({
  open,
  selectedCount,
  reviewCount,
  skippedFailedCount,
  source,
  localOnly,
  onClose,
  onConfirm,
}: ApplyConfirmDialogProps) {
  const nameLabel = selectedCount === 1 ? "name" : "names";

  return (
    <Modal
      footer={
        <>
          <Button onClick={onClose}>Keep reviewing</Button>
          <Button onClick={onConfirm} variant="danger">
            Rename files
          </Button>
        </>
      }
      onClose={onClose}
      open={open}
      title={`Write ${selectedCount} selected ${nameLabel}?`}
    >
      <div className="confirm-list">
        <p>
          Folionym will rename only the checked files to their exact reviewed targets. Changed sources and new
          collisions fail safely.
        </p>
        <div>
          <strong>{selectedCount}</strong>
          <span>selected</span>
        </div>
        <div>
          <strong>{reviewCount}</strong>
          <span>need review</span>
        </div>
        <div>
          <strong>{skippedFailedCount}</strong>
          <span>not applicable</span>
        </div>
        <div className="confirm-facts">
          <div>
            <span>Scope</span>
            <code>{compactPath(source)}</code>
          </div>
          <div>
            <span>Mode</span>
            <strong>Exact targets · no recompute</strong>
          </div>
          <div>
            <span>Privacy</span>
            <strong>{localOnly ? "Local only" : "External model"}</strong>
          </div>
        </div>
      </div>
    </Modal>
  );
}
