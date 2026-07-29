export function Switch({
  checked,
  label,
  description,
  onChange,
}: {
  checked: boolean;
  label: string;
  description?: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="switch-row">
      <span>
        <span className="switch-row__label">{label}</span>
        {description && <span className="switch-row__description">{description}</span>}
      </span>
      <input
        checked={checked}
        className="switch-input"
        onChange={(event) => {
          onChange(event.target.checked);
        }}
        type="checkbox"
      />
      <span aria-hidden="true" className="switch-control" />
    </label>
  );
}
