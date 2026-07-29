import { compactPath } from "../../components";
import type { Plan, PreviewStatus } from "../../types";

function filterLabel(status: PreviewStatus | "all"): string {
  switch (status) {
    case "ready":
      return "Ready";
    case "review":
      return "Review";
    case "skipped":
      return "Skipped";
    case "failed":
      return "Failed";
    default:
      return "All";
  }
}

const filterStatuses: Array<PreviewStatus | "all"> = ["all", "ready", "review", "skipped", "failed"];

export type ProcessingFact = { label: string; value: string };

type FilterRailProps = {
  plan: Plan;
  filter: PreviewStatus | "all";
  selectedCount: number;
  facts: ProcessingFact[];
  onFilterChange: (filter: PreviewStatus | "all") => void;
  onChangeSource: () => void;
};

export function FilterRail({
  plan,
  filter,
  selectedCount,
  facts,
  onFilterChange,
  onChangeSource,
}: FilterRailProps) {
  const skippedFailed = plan.counts.skipped + plan.counts.failed;

  return (
    <aside className="filter-rail" aria-label="Filters and selection">
      <section className="rail-block">
        <h2 className="rail-title">Review set</h2>
        <p className="rail-lede">Checked rows keep their exact targets. Apply does not recompute names.</p>
        <dl className="metric-list">
          <div>
            <dt>Selected</dt>
            <dd>{selectedCount}</dd>
          </div>
          <div>
            <dt>Ready</dt>
            <dd>{plan.counts.ready}</dd>
          </div>
          <div>
            <dt>Needs review</dt>
            <dd>{plan.counts.review}</dd>
          </div>
          <div>
            <dt>Skipped / failed</dt>
            <dd>{skippedFailed}</dd>
          </div>
        </dl>
      </section>
      <section className="rail-block">
        <h2 className="rail-title" id="filter-heading">
          Show
        </h2>
        <div className="filter-list" role="tablist" aria-labelledby="filter-heading">
          {filterStatuses.map((status) => (
            <button
              aria-selected={filter === status}
              className={`filter ${filter === status ? "is-active" : ""}`}
              key={status}
              onClick={() => {
                onFilterChange(status);
              }}
              role="tab"
              type="button"
            >
              {filterLabel(status)} <span>{status === "all" ? plan.counts.all : plan.counts[status]}</span>
            </button>
          ))}
        </div>
      </section>
      <section className="rail-block rail-block--quiet">
        <h2 className="rail-title">Processing</h2>
        <ul className="fact-list">
          {facts.map((fact) => (
            <li key={fact.label}>
              <span>{fact.label}</span>
              <strong>{fact.value}</strong>
            </li>
          ))}
        </ul>
        <p className="rail-source">
          <span>Source</span>
          <strong title={plan.source}>{compactPath(plan.source)}</strong>
        </p>
        <button className="linkish" onClick={onChangeSource} type="button">
          Change source
        </button>
      </section>
    </aside>
  );
}
