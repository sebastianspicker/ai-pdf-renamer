const STAGES = [
  { id: 1 as const, key: "source", label: "Source", detail: "Folder chosen" },
  { id: 2 as const, key: "preview", label: "Preview", detail: "Review names" },
  { id: 3 as const, key: "apply", label: "Apply", detail: "Write files" },
];

export function StageSpine({ active }: { active: 1 | 2 | 3 }) {
  return (
    <nav aria-label="Rename stages" className="stage-spine">
      <ol>
        {STAGES.map((stage) => {
          const done = active > stage.id;
          const current = active === stage.id;
          const upcoming = active < stage.id;
          const className = [
            "stage",
            done ? "stage--done" : "",
            current ? "stage--current" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <li
              aria-current={current ? "step" : undefined}
              className={className}
              key={stage.id}
            >
              <button
                className="stage-hit"
                data-stage={stage.key}
                disabled={upcoming}
                type="button"
              >
                <span aria-hidden="true" className="stage-node" />
                <span className="stage-label">
                  <strong>{stage.label}</strong>
                  <em>{stage.detail}</em>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
