import { useState } from "react";
import { StatusPill } from "../../components";
import { ArrowRightIcon, DocumentIcon } from "../../icons";
import type { ApplyItem, ApplyStatus } from "../../types";

const reportFilters: Array<ApplyStatus | "all"> = ["all", "renamed", "unchanged", "failed", "cancelled"];

export function ApplyReportLedger({ items }: { items: ApplyItem[] }) {
  const [filter, setFilter] = useState<ApplyStatus | "all">("all");
  const visibleItems = items.filter((item) => filter === "all" || item.status === filter);

  return (
    <section className="result-list">
      <header>
        <div>
          <span className="eyebrow">Ledger</span>
          <h2>Per-file outcome</h2>
        </div>
        <div className="result-filters">
          {reportFilters.map((status) => (
            <button
              className={filter === status ? "is-active" : ""}
              key={status}
              onClick={() => {
                setFilter(status);
              }}
              type="button"
            >
              {status}
            </button>
          ))}
        </div>
      </header>
      <div className="result-table">
        {visibleItems.map((item) => (
          <div className="result-row" key={item.item_id}>
            <DocumentIcon />
            <span>{item.source_name}</span>
            <ArrowRightIcon />
            <strong>{item.target_name ?? "No target"}</strong>
            <StatusPill status={item.status === "skipped" ? "unchanged" : item.status} />
            {item.reason && <small>{item.reason}</small>}
          </div>
        ))}
      </div>
    </section>
  );
}
