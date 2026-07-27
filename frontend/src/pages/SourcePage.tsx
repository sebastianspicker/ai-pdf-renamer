import { useCallback, useState } from "react";
import { ApiError, api, errorMessage } from "../api";
import {
  AppChrome,
  Button,
  ErrorBanner,
  Field,
  FineTuneDrawer,
  FolderBrowser,
  Modal,
  RunOverlay,
  TextInput,
} from "../components";
import { useRun } from "../hooks/useRun";
import {
  ArrowRightIcon,
  DocumentIcon,
  FolderIcon,
  InfoIcon,
  SlidersIcon,
  WarningIcon,
} from "../icons";
import { isLocalEndpoint } from "../lib/privacy";
import { navigate } from "../lib/routing";
import type { Bootstrap, Run, Settings } from "../types";

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function SourcePage({
  bootstrap,
  onBootstrapChange,
}: {
  bootstrap: Bootstrap;
  onBootstrapChange: (bootstrap: Bootstrap) => void;
}) {
  const [settings, setSettings] = useState(bootstrap.settings);
  const [kind, setKind] = useState<"directory" | "file">(bootstrap.settings.single_file ? "file" : "directory");
  const [path, setPath] = useState(bootstrap.settings.single_file || bootstrap.settings.directory);
  const [folderOpen, setFolderOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [startError, setStartError] = useState("");
  const [ack, setAck] = useState<{ endpoint: string } | null>(null);

  const updateSetting = (key: keyof Settings, value: Settings[keyof Settings]) => {
    setSettings((current) => ({ ...current, [key]: value }));
  };

  const onRunComplete = useCallback((run: Run) => {
    if (run.state === "completed" && run.plan_id) {
      sessionStorage.setItem("folionym.plan", run.plan_id);
      sessionStorage.removeItem("folionym.report");
      navigate("preview");
    }
  }, []);

  const { run, error: runError, setError: setRunError } = useRun(runId, onRunComplete);

  const startPreview = async (acknowledge = false) => {
    setStartError("");
    if (!path.trim()) {
      setStartError(`Choose a ${kind === "file" ? "PDF" : "folder"} before continuing.`);
      return;
    }
    try {
      const result = await api.startPreview(kind, path.trim(), settings, acknowledge);
      setAck(null);
      setRunId(result.run_id);
      onBootstrapChange({ ...bootstrap, settings });
    } catch (requestError) {
      if (
        requestError instanceof ApiError &&
        requestError.status === 409 &&
        typeof requestError.detail === "object" &&
        requestError.detail !== null &&
        "code" in requestError.detail &&
        "endpoint" in requestError.detail &&
        requestError.detail.code === "external_endpoint_ack_required"
      ) {
        setAck({ endpoint: String(requestError.detail.endpoint ?? settings.llm_url) });
        return;
      }
      setStartError(errorMessage(requestError));
    }
  };

  const localOnly = isLocalEndpoint(settings);

  return (
    <AppChrome
      active={1}
      external={!localOnly}
      overlays={
        <>
          <FolderBrowser
            bootstrap={bootstrap}
            initialPath={path || settings.directory}
            onChoose={(chosen) => {
              setPath(chosen);
              setSettings((current) => ({ ...current, directory: chosen }));
              setFolderOpen(false);
            }}
            onClose={() => setFolderOpen(false)}
            open={folderOpen}
          />
          <FineTuneDrawer
            onChange={updateSetting}
            onClose={() => setDrawerOpen(false)}
            open={drawerOpen}
            settings={settings}
          />
          <Modal
            footer={
              <>
                <Button onClick={() => setAck(null)}>Cancel</Button>
                <Button onClick={() => void startPreview(true)} variant="primary">
                  Continue once
                </Button>
              </>
            }
            onClose={() => setAck(null)}
            open={Boolean(ack)}
            title="Confirm external endpoint"
          >
            <div className="consent">
              <WarningIcon />
              <p>
                Document-derived text may leave this machine for <strong>{ack?.endpoint}</strong>. Acknowledgement
                applies only to this exact endpoint for this session.
              </p>
            </div>
          </Modal>
          <RunOverlay
            error={runError}
            onCancel={() => runId && void api.cancel(runId)}
            onClose={() => {
              setRunId(null);
              setRunError("");
            }}
            run={run}
          />
        </>
      }
      source={path}
    >
      <main className="source-shell">
        <section className="source-main">
          <div className="page-heading">
            <div>
              <span className="eyebrow">Source</span>
              <h1>Choose local PDFs</h1>
              <p>
                Select a folder or one file. Preview proposes names; nothing is written until you apply exact
                targets.
              </p>
            </div>
            <Button onClick={() => setDrawerOpen(true)}>
              <SlidersIcon /> Fine-tune
            </Button>
          </div>
          {startError && <ErrorBanner message={startError} onDismiss={() => setStartError("")} />}
          <div className="source-card">
            <div aria-label="Source type" className="segmented" role="group">
              <button
                aria-pressed={kind === "directory"}
                className={kind === "directory" ? "is-active" : ""}
                onClick={() => {
                  setKind("directory");
                  setPath(settings.directory);
                }}
              >
                Folder
              </button>
              <button
                aria-pressed={kind === "file"}
                className={kind === "file" ? "is-active" : ""}
                onClick={() => {
                  setKind("file");
                  setPath(settings.single_file);
                }}
              >
                Single PDF
              </button>
            </div>
            <div className="source-picker">
              <div className="source-picker__icon">
                {kind === "directory" ? <FolderIcon size={28} /> : <DocumentIcon size={28} />}
              </div>
              <div className="source-picker__copy">
                <strong>{path || (kind === "directory" ? "No folder selected" : "No PDF selected")}</strong>
                <span>
                  {kind === "directory"
                    ? "PDFs only. Subfolders follow the recursive setting."
                    : "Absolute path to one local PDF."}
                </span>
              </div>
              {kind === "directory" ? (
                <Button onClick={() => setFolderOpen(true)}>Browse</Button>
              ) : (
                <Button onClick={() => setPath("")}>Clear</Button>
              )}
            </div>
            {kind === "file" && (
              <Field className="source-file-field" label="Absolute PDF path">
                <TextInput
                  onChange={(event) => setPath(event.target.value)}
                  placeholder="/Users/you/Documents/scan.pdf"
                  value={path}
                />
              </Field>
            )}
          </div>
          <section className="run-summary">
            <div className="run-summary__header">
              <div>
                <span className="eyebrow">Configuration</span>
                <h2>Effective settings</h2>
              </div>
              <button className="text-button" onClick={() => setDrawerOpen(true)}>
                Edit
              </button>
            </div>
            <div className="summary-list">
              <SummaryRow label="Filename style" value={`${settings.case} · ${settings.date_format.toUpperCase()} date`} />
              <SummaryRow
                label="Extraction"
                value={`${settings.use_ocr ? "OCR" : "Text"} · ${settings.recursive ? "include subfolders" : "this folder only"}`}
              />
              <SummaryRow
                label="Enrichment"
                value={
                  settings.use_llm
                    ? `${settings.llm_model || "Configured model"} · ${localOnly ? "local endpoint" : "external endpoint"}`
                    : "Rules and heuristics only"
                }
              />
              <SummaryRow label="Apply" value="Reviewed exact names only" />
            </div>
          </section>
          <div className="source-notice">
            <InfoIcon />
            <p>
              <strong>Preview does not rename files.</strong> Apply re-checks each source fingerprint and target
              before writing.
            </p>
          </div>
        </section>
        <aside className="source-aside">
          <span className="eyebrow">Next</span>
          <h2>Build a preview</h2>
          <p>Extract naming evidence and rank proposals. You review every name before any path changes.</p>
          <div className="aside-checks">
            <span>
              <i>
                <span />
              </i>
              Sources stay in place during Preview
            </span>
            <span>
              <i>
                <span />
              </i>
              Uncertain items are marked Review
            </span>
            <span>
              <i>
                <span />
              </i>
              Apply uses only checked exact targets
            </span>
          </div>
          <Button className="button--full" onClick={() => void startPreview()} variant="primary">
            Build preview <ArrowRightIcon />
          </Button>
        </aside>
      </main>
    </AppChrome>
  );
}
