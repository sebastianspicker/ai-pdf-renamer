import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage } from "../api";
import { AppChrome, Button, Modal, PageLoader, RunOverlay, compactPath } from "../components";
import { useRun } from "../hooks/useRun";
import { WarningIcon } from "../icons";
import { isLocalEndpoint } from "../lib/privacy";
import { navigate } from "../lib/routing";
import type { Bootstrap, Plan, PreviewStatus, Run, Settings } from "../types";
import { FilterRail } from "./preview/FilterRail";
import { Inspector } from "./preview/Inspector";
import { Ledger } from "./preview/Ledger";

function caseLabel(value: string): string {
  if (value === "kebabCase") return "kebab";
  if (value === "snake_case") return "snake";
  if (value === "camelCase") return "camel";
  if (value === "Title Case") return "title";
  return value;
}

function dateLabel(value: string): string {
  if (value === "ymd") return "YYYYMMDD";
  if (value === "dmy") return "DDMMYYYY";
  if (value === "mdy") return "MMDDYYYY";
  return value.toUpperCase();
}

function processingFacts(settings: Settings) {
  return [
    { label: "Naming", value: `${caseLabel(settings.case)} · ${dateLabel(settings.date_format)}` },
    { label: "Extraction", value: settings.use_ocr ? "OCR" : "Text" },
    {
      label: "Enrichment",
      value: settings.use_llm ? settings.llm_model || "Model" : "Rules only",
    },
    { label: "Workers", value: settings.workers || "1" },
  ];
}

export function PreviewPage({ bootstrap }: { bootstrap: Bootstrap }) {
  const planId = sessionStorage.getItem("folionym.plan");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<PreviewStatus | "all">("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeId, setActiveId] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);

  useEffect(() => {
    if (!planId) {
      navigate("source");
      return;
    }
    let active = true;
    const loadPlan = async () => {
      try {
        const next = await api.plan(planId);
        if (!active) return;
        setPlan(next);
        setSelected(new Set(next.items.filter((item) => item.included).map((item) => item.id)));
        setActiveId(next.items[0]?.id ?? null);
      } catch (requestError: unknown) {
        if (active) setError(errorMessage(requestError));
      } finally {
        if (active) setLoading(false);
      }
    };
    void loadPlan();
    return () => {
      active = false;
    };
  }, [planId]);

  const onRunComplete = useCallback((run: Run) => {
    if (run.state === "completed" && run.report_id) {
      sessionStorage.setItem("folionym.report", run.report_id);
      navigate("apply");
    }
  }, []);
  const { run, error: runError, setError: setRunError } = useRun(runId, onRunComplete);

  const visibleItems = useMemo(() => {
    if (!plan) return [];
    const normalized = query.trim().toLowerCase();
    return plan.items.filter((item) => {
      if (filter !== "all" && item.status !== filter) return false;
      return (
        !normalized ||
        item.current_name.toLowerCase().includes(normalized) ||
        item.proposed_name?.toLowerCase().includes(normalized)
      );
    });
  }, [plan, filter, query]);

  const activeItem = plan?.items.find((item) => item.id === activeId) ?? null;
  const selectable = visibleItems.filter((item) => item.status === "ready" || item.status === "review");
  const allVisibleSelected = selectable.length > 0 && selectable.every((item) => selected.has(item.id));

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectVisible = () => {
    setSelected((current) => {
      const next = new Set(current);
      selectable.forEach((item) => next.add(item.id));
      return next;
    });
  };

  const clearSelection = () => {
    setSelected(new Set());
  };

  const apply = async () => {
    if (!plan) return;
    try {
      const result = await api.apply(plan.id, plan.revision, [...selected]);
      setConfirmOpen(false);
      setRunId(result.run_id);
    } catch (requestError: unknown) {
      setError(errorMessage(requestError));
      setConfirmOpen(false);
    }
  };

  if (loading) return <PageLoader label="Loading preview" />;
  if (!plan || error) {
    return (
      <div className="fatal-state">
        <WarningIcon size={28} />
        <h1>Preview unavailable</h1>
        <p>{error || "This preview is no longer available."}</p>
        <Button onClick={() => {
          navigate("source");
        }}>Start again</Button>
      </div>
    );
  }

  const localOnly = isLocalEndpoint(bootstrap.settings);
  const facts = processingFacts(bootstrap.settings);
  const skippedFailed = plan.counts.skipped + plan.counts.failed;

  return (
    <AppChrome
      active={2}
      external={!localOnly}
      footer={
        <>
          <div className="consequence-copy">
            <strong>
              {selected.size} exact name{selected.size === 1 ? "" : "s"} ready to write
            </strong>
            <p>
              Apply re-checks each source fingerprint and target collision. Files outside the selection stay untouched.
            </p>
          </div>
          <div className="consequence-actions">
            <Button onClick={() => {
              navigate("source");
            }}>Back to Source</Button>
            <Button disabled={selected.size === 0} onClick={() => {
              setConfirmOpen(true);
            }} variant="primary">
              Apply {selected.size} name{selected.size === 1 ? "" : "s"}
            </Button>
          </div>
        </>
      }
      overlays={
        <>
          <Modal
            footer={
              <>
                <Button onClick={() => {
                  setConfirmOpen(false);
                }}>Keep reviewing</Button>
                <Button onClick={() => {
                  void apply();
                }} variant="danger">
                  Rename files
                </Button>
              </>
            }
            onClose={() => {
              setConfirmOpen(false);
            }}
            open={confirmOpen}
            title={`Write ${selected.size} selected name${selected.size === 1 ? "" : "s"}?`}
          >
            <div className="confirm-list">
              <p>
                Folionym will rename only the checked files to their exact reviewed targets. Changed sources and new
                collisions fail safely.
              </p>
              <div>
                <strong>{selected.size}</strong>
                <span>selected</span>
              </div>
              <div>
                <strong>{plan.counts.review}</strong>
                <span>need review</span>
              </div>
              <div>
                <strong>{skippedFailed}</strong>
                <span>not applicable</span>
              </div>
              <div className="confirm-facts">
                <div>
                  <span>Scope</span>
                  <code>{compactPath(plan.source)}</code>
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
          <RunOverlay
            error={runError}
            onCancel={() => {
              if (runId) void api.cancel(runId);
            }}
            onClose={() => {
              setRunId(null);
              setRunError("");
            }}
            run={run}
          />
        </>
      }
      source={plan.source}
      sourceMeta={`${plan.counts.all} items`}
    >
      <>
        <FilterRail
          facts={facts}
          filter={filter}
          onChangeSource={() => {
            navigate("source");
          }}
          onFilterChange={setFilter}
          plan={plan}
          selectedCount={selected.size}
        />
        <Ledger
          activeId={activeId}
          allVisibleSelected={allVisibleSelected}
          error={error}
          onActivate={setActiveId}
          onClear={clearSelection}
          onDismissError={() => {
            setError("");
          }}
          onQueryChange={setQuery}
          onSelectVisible={selectVisible}
          onToggle={toggle}
          plan={plan}
          query={query}
          selectableCount={selectable.length}
          selected={selected}
          visibleItems={visibleItems}
        />
        <Inspector item={activeItem} plan={plan} />
      </>
    </AppChrome>
  );
}
