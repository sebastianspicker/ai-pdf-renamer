import { StatusPill, formatBytes } from "../../components";
import { DocumentIcon } from "../../icons";
import type { Plan, PreviewItem } from "../../types";

export function Inspector({ item, plan }: { item: PreviewItem | null; plan: Plan }) {
  if (!item) {
    return (
      <aside className="inspector inspector--empty" aria-label="Selected document evidence">
        <DocumentIcon size={28} />
        <p>Select a document to inspect its naming evidence.</p>
      </aside>
    );
  }

  const metadataEntries = Object.entries(item.metadata)
    .filter(([, value]) => value !== null && value !== "" && typeof value !== "object")
    .slice(0, 8);

  return (
    <aside className="inspector" aria-label="Selected document evidence">
      <div className="inspector-head">
        <p className="kicker">Evidence</p>
        <h2>Selected document</h2>
        <StatusPill status={item.status} />
      </div>
      <figure className="thumb">
        <div className="thumb-frame">
          <img
            alt={`First page of ${item.current_name}`}
            src={`/api/v1/plans/${plan.id}/items/${item.id}/thumbnail`}
          />
        </div>
        <figcaption>First page</figcaption>
      </figure>
      <dl className="evidence">
        <div>
          <dt>Current</dt>
          <dd>
            <code>{item.current_name}</code>
          </dd>
        </div>
        <div>
          <dt>Proposed</dt>
          <dd>
            <code>{item.proposed_name ?? "No proposal"}</code>
          </dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{formatBytes(item.size)}</dd>
        </div>
        <div>
          <dt>Modified</dt>
          <dd>{item.modified_at ? new Date(item.modified_at).toLocaleString() : "Unavailable"}</dd>
        </div>
        {metadataEntries.map(([key, value]) => (
          <div key={key}>
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd>{String(value)}</dd>
          </div>
        ))}
      </dl>
      {item.reason && (
        <div className={`inspector-note item-reason item-reason--${item.status}`}>
          <strong>Why this name</strong>
          <p>{item.reason}</p>
        </div>
      )}
    </aside>
  );
}
