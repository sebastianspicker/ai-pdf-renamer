export type StatusPillStatus =
  | "ready"
  | "review"
  | "skipped"
  | "failed"
  | "renamed"
  | "unchanged"
  | "cancelled";

function sentenceCase(value: string): string {
  if (!value) return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function StatusPill({
  status,
  label,
}: {
  status: StatusPillStatus;
  label?: string;
}) {
  return <span className={`status status--${status}`}>{label ?? sentenceCase(status)}</span>;
}
