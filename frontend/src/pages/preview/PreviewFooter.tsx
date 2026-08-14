import { Button } from "../../components";

type PreviewFooterProps = {
  selectedCount: number;
  onBack: () => void;
  onOpenConfirm: () => void;
};

export function PreviewFooter({ selectedCount, onBack, onOpenConfirm }: PreviewFooterProps) {
  const nameLabel = selectedCount === 1 ? "name" : "names";

  return (
    <>
      <div className="consequence-copy">
        <strong>
          {selectedCount} exact {nameLabel} ready to write
        </strong>
        <p>Apply re-checks each source fingerprint and target collision. Files outside the selection stay untouched.</p>
      </div>
      <div className="consequence-actions">
        <Button onClick={onBack}>Back to Source</Button>
        <Button disabled={selectedCount === 0} onClick={onOpenConfirm} variant="primary">
          Apply {selectedCount} {nameLabel}
        </Button>
      </div>
    </>
  );
}
