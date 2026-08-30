import { AppChrome, Button, PageLoader, RunOverlay } from "../components";
import { WarningIcon } from "../icons";
import { isLocalEndpoint } from "../lib/privacy";
import { navigate } from "../lib/routing";
import type { Bootstrap, Settings } from "../types";
import { ApplyConfirmDialog } from "./preview/ApplyConfirmDialog";
import { FilterRail } from "./preview/FilterRail";
import { Inspector } from "./preview/Inspector";
import { Ledger } from "./preview/Ledger";
import { PreviewFooter } from "./preview/PreviewFooter";
import { usePreviewApply } from "./preview/usePreviewApply";
import { usePreviewPlan } from "./preview/usePreviewPlan";

function caseLabel(value: string): string {
  if (value === "kebabCase") return "kebab";
  if (value === "snakeCase") return "snake";
  if (value === "camelCase") return "camel";
  return value;
}

function dateLabel(value: string): string {
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
  const preview = usePreviewPlan(planId);
  const apply = usePreviewApply({
    plan: preview.plan,
    selected: preview.selected,
    setError: preview.setError,
  });

  if (preview.loading) return <PageLoader label="Loading preview" />;
  if (!preview.plan || preview.error) {
    return (
      <div className="fatal-state">
        <WarningIcon size={28} />
        <h1>Preview unavailable</h1>
        <p>{preview.error || "This preview is no longer available."}</p>
        <Button onClick={() => {
          navigate("source");
        }}>Start again</Button>
      </div>
    );
  }

  const localOnly = isLocalEndpoint(bootstrap.settings);
  const facts = processingFacts(bootstrap.settings);
  return (
    <AppChrome
      active={2}
      external={!localOnly}
      footer={
        <PreviewFooter
          onBack={() => {
            navigate("source");
          }}
          onOpenConfirm={apply.openConfirm}
          selectedCount={preview.selected.size}
        />
      }
      overlays={
        <>
          <ApplyConfirmDialog
            localOnly={localOnly}
            onClose={apply.closeConfirm}
            onConfirm={() => {
              void apply.apply();
            }}
            open={apply.confirmOpen}
            reviewCount={preview.plan.counts.review}
            selectedCount={preview.selected.size}
            skippedFailedCount={preview.plan.counts.skipped + preview.plan.counts.failed}
            source={preview.plan.source}
          />
          <RunOverlay
            error={apply.runError}
            onCancel={apply.cancelRun}
            onClose={apply.closeRun}
            run={apply.run}
          />
        </>
      }
      source={preview.plan.source}
      sourceMeta={`${preview.plan.counts.all} items`}
    >
      <>
        <FilterRail
          facts={facts}
          filter={preview.filter}
          onChangeSource={() => {
            navigate("source");
          }}
          onFilterChange={preview.setFilter}
          plan={preview.plan}
          selectedCount={preview.selected.size}
        />
        <Ledger
          activeId={preview.activeId}
          allVisibleSelected={preview.allVisibleSelected}
          error={preview.error}
          onActivate={preview.setActiveId}
          onClear={() => {
            preview.setSelected(new Set());
          }}
          onDismissError={() => {
            preview.setError("");
          }}
          onQueryChange={preview.setQuery}
          onSelectVisible={preview.selectVisible}
          onToggle={preview.toggle}
          plan={preview.plan}
          query={preview.query}
          selectableCount={preview.selectableCount}
          selected={preview.selected}
          visibleItems={preview.visibleItems}
        />
        <Inspector item={preview.activeItem} plan={preview.plan} />
      </>
    </AppChrome>
  );
}
