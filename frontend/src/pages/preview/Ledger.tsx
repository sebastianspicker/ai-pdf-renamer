import { Button, Checkbox, ErrorBanner, StatusPill } from "../../components";
import { SearchIcon } from "../../icons";
import type { Plan, PreviewItem, PreviewStatus } from "../../types";

function filterLabel(status: PreviewStatus): string {
  switch (status) {
    case "ready":
      return "Ready";
    case "review":
      return "Review";
    case "skipped":
      return "Skipped";
    default:
      return "Failed";
  }
}

type LedgerProps = {
  plan: Plan;
  visibleItems: PreviewItem[];
  selected: Set<string>;
  activeId: string | null;
  query: string;
  error: string;
  allVisibleSelected: boolean;
  selectableCount: number;
  onQueryChange: (query: string) => void;
  onToggle: (id: string) => void;
  onSelectVisible: () => void;
  onClear: () => void;
  onActivate: (id: string) => void;
  onDismissError: () => void;
};

export function Ledger({
  visibleItems,
  selected,
  activeId,
  query,
  error,
  allVisibleSelected,
  selectableCount,
  onQueryChange,
  onToggle,
  onSelectVisible,
  onClear,
  onActivate,
  onDismissError,
}: LedgerProps) {
  return (
    <section className="ledger-panel" id="ledger">
      <div className="ledger-head">
        <div>
          <p className="kicker">Preview · immutable plan once applied</p>
          <h1>Rename ledger</h1>
          <p className="lede">
            Compare source to proposed name. Checked rows keep their exact target on Apply.
          </p>
        </div>
        <div className="ledger-tools">
          <label className="search">
            <span className="visually-hidden">Search filenames</span>
            <SearchIcon />
            <input
              aria-label="Search filenames"
              autoComplete="off"
              onChange={(event) => {
                onQueryChange(event.target.value);
              }}
              placeholder="Filter by name…"
              type="search"
              value={query}
            />
          </label>
          <div className="tool-cluster">
            <Button
              disabled={selectableCount === 0 || allVisibleSelected}
              onClick={onSelectVisible}
              type="button"
            >
              Select visible
            </Button>
            <Button disabled={selected.size === 0} onClick={onClear} type="button">
              Clear
            </Button>
          </div>
        </div>
      </div>

      {error && <ErrorBanner message={error} onDismiss={onDismissError} />}

      <div className="ledger-chrome" aria-hidden="true">
        <span className="col-check" />
        <span className="col-from">Current</span>
        <span className="col-arrow" />
        <span className="col-to">Proposed</span>
        <span className="col-status">Status</span>
      </div>

      <div
        aria-label="Proposed renames"
        aria-multiselectable="true"
        className="ledger"
        id="ledger-list"
        role="listbox"
      >
        {visibleItems.map((item) => {
          const isSelectable = item.status === "ready" || item.status === "review";
          const isSelected = selected.has(item.id);
          const isActive = activeId === item.id;
          const mutedTarget = !item.proposed_name || item.status === "skipped" || item.status === "failed";
          return (
            <article
              aria-selected={isSelected}
              className={`row ${isSelected ? "is-selected" : ""} ${isActive ? "is-active" : ""}`}
              data-id={item.id}
              data-status={item.status}
              key={item.id}
              onClick={() => {
                onActivate(item.id);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onActivate(item.id);
                }
              }}
              role="option"
              tabIndex={isActive ? 0 : -1}
            >
              <label className="row-check" onClick={(event) => {
                event.stopPropagation();
              }}>
                <Checkbox
                  aria-label={`Include ${item.proposed_name ?? item.current_name}`}
                  checked={isSelected}
                  disabled={!isSelectable}
                  onChange={() => {
                    onToggle(item.id);
                  }}
                />
                <span className="visually-hidden">Include</span>
              </label>
              <div className="name-pair">
                <code className="name name--from" title={item.current_name}>
                  {item.current_name}
                </code>
                <span aria-hidden="true" className="arrow">
                  →
                </span>
                <code
                  className={`name name--to ${mutedTarget ? "name--muted" : ""}`}
                  title={item.proposed_name ?? ""}
                >
                  {item.proposed_name ??
                    (item.status === "skipped"
                      ? "Unchanged · already named"
                      : item.status === "failed"
                        ? "No proposal · extract failed"
                        : "No proposal")}
                </code>
              </div>
              <StatusPill label={filterLabel(item.status)} status={item.status} />
            </article>
          );
        })}
        {visibleItems.length === 0 && (
          <div className="table-empty">
            <SearchIcon />
            <strong>No matching documents</strong>
            <span>Clear the search or choose another status.</span>
          </div>
        )}
      </div>
    </section>
  );
}
