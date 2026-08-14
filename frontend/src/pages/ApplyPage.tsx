import { useEffect, useState } from "react";
import { api, errorMessage } from "../api";
import { AppChrome, Button, PageLoader } from "../components";
import { ArrowRightIcon, InfoIcon, WarningIcon } from "../icons";
import { isLocalEndpoint } from "../lib/privacy";
import { navigate } from "../lib/routing";
import type { Bootstrap, Report } from "../types";
import { ApplyReportLedger } from "./apply/ApplyReportLedger";

function ResultMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "success" | "danger";
}) {
  return (
    <div className={`result-metric ${tone ? `result-metric--${tone}` : ""}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

export function ApplyPage({ bootstrap }: { bootstrap: Bootstrap }) {
  const reportId = sessionStorage.getItem("folionym.report");
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!reportId) {
      navigate("source");
      return;
    }
    const loadReport = async () => {
      try {
        setReport(await api.report(reportId));
      } catch (requestError: unknown) {
        setError(errorMessage(requestError));
      }
    };
    void loadReport();
  }, [reportId]);

  if (!report && !error) return <PageLoader label="Loading report" />;
  if (!report) {
    return (
      <div className="fatal-state">
        <WarningIcon />
        <h1>Report unavailable</h1>
        <p>{error}</p>
        <Button onClick={() => {
          navigate("source");
        }}>Back to Source</Button>
      </div>
    );
  }

  const changed = report.counts.renamed;
  const hasFailures = report.counts.failed + report.counts.cancelled > 0;
  const localOnly = isLocalEndpoint(bootstrap.settings);

  return (
    <AppChrome
      active={3}
      external={!localOnly}
      source={report.source}
      sourceMeta={`${report.items.length} items`}
    >
      <main className="result-shell">
        <header className="result-heading">
          <div className={`result-mark ${hasFailures ? "result-mark--partial" : ""}`}>
            {hasFailures ? <WarningIcon size={26} /> : <span>✓</span>}
          </div>
          <div>
            <span className="eyebrow">Apply</span>
            <h1>
              {hasFailures
                ? "Run finished with issues"
                : `${changed} file${changed === 1 ? "" : "s"} renamed`}
            </h1>
            <p>
              {hasFailures
                ? "Validated renames were written. Items that failed checks were left unchanged."
                : "Each selected file passed fingerprint and collision checks, then received its exact reviewed name."}
            </p>
          </div>
          <Button onClick={() => {
            navigate("source");
          }} variant="primary">
            New source <ArrowRightIcon />
          </Button>
        </header>
        <section aria-label="Apply totals" className="result-overview">
          <ResultMetric label="Renamed" tone="success" value={report.counts.renamed} />
          <ResultMetric label="Unchanged" value={report.counts.unchanged} />
          <ResultMetric
            label="Failed"
            tone={report.counts.failed ? "danger" : undefined}
            value={report.counts.failed}
          />
          <ResultMetric label="Cancelled" value={report.counts.cancelled} />
        </section>
        <ApplyReportLedger items={report.items} />
        <div className="source-notice">
          <InfoIcon />
          <p>
            This report covers the current local session only. Use the CLI for watch mode, diagnostics, or undo
            from a rename log.
          </p>
        </div>
      </main>
    </AppChrome>
  );
}
