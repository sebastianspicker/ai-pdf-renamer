import type { Bootstrap, DirectoryListing, Plan, Report, Run, Settings } from "./types";

const source = "/demo/inbox";
const planId = "demo-plan";
const reportId = "demo-report";

const settings: Settings = {
  directory: source,
  single_file: "",
  language: "en",
  case: "kebabCase",
  date_format: "ymd",
  preset: "high-confidence-heuristic",
  project: "",
  version: "",
  template: "{date}-{category}-{keywords}-{summary}",
  backup_dir: "",
  rename_log: "",
  export_metadata: "",
  summary_json: "",
  rules_file: "",
  post_rename_hook: "",
  llm_url: "http://127.0.0.1:11434/v1/completions",
  llm_model: "qwen2.5:3b",
  llm_timeout: "30",
  max_tokens: "",
  max_content_chars: "",
  max_content_tokens: "",
  workers: "2",
  max_filename_chars: "180",
  dry_run: true,
  use_llm: false,
  use_ocr: false,
  recursive: false,
  skip_already_named: true,
  use_pdf_metadata_date: true,
  use_structured_fields: true,
  write_pdf_metadata: false,
  use_vision_fallback: false,
  simple_naming_mode: false,
  vision_first: false,
  acknowledged_external_endpoint: "",
};

const bootstrap: Bootstrap = {
  settings,
  roots: [{ name: "Demo inbox", path: source, pdf_count: 6 }],
  capabilities: { pdf: true, ocr: false, vision: false, llm: false },
};

const plan: Plan = {
  id: planId,
  revision: 1,
  source,
  source_kind: "directory",
  created_at: "2026-07-29T09:00:00Z",
  counts: { all: 6, ready: 4, review: 1, skipped: 1, failed: 0 },
  items: [
    {
      id: "invoice",
      current_name: "scan_0042.pdf",
      source_path: `${source}/scan_0042.pdf`,
      proposed_name: "20260712-invoice-office-supplies.pdf",
      status: "ready",
      included: true,
      reason: "Invoice date and category were found in extracted text.",
      size: 184320,
      modified_at: "2026-07-28T14:20:00Z",
      metadata: { category: "invoice", invoice_id: "INV-1042", amount: "148.60 EUR" },
    },
    {
      id: "minutes",
      current_name: "meeting-notes-final.pdf",
      source_path: `${source}/meeting-notes-final.pdf`,
      proposed_name: "20260718-minutes-project-kite.pdf",
      status: "ready",
      included: true,
      reason: "Document title and creation date matched the minutes rule.",
      size: 96256,
      modified_at: "2026-07-27T08:40:00Z",
      metadata: { category: "minutes", project: "Project Kite", pages: 3 },
    },
    {
      id: "statement",
      current_name: "document-2026-07.pdf",
      source_path: `${source}/document-2026-07.pdf`,
      proposed_name: "20260701-statement-account-summary.pdf",
      status: "review",
      included: true,
      reason: "The category is plausible, but the document date should be reviewed.",
      size: 421888,
      modified_at: "2026-07-26T16:10:00Z",
      metadata: { category: "statement", pages: 8 },
    },
    {
      id: "receipt",
      current_name: "receipt-cafe.pdf",
      source_path: `${source}/receipt-cafe.pdf`,
      proposed_name: "20260722-receipt-team-lunch.pdf",
      status: "ready",
      included: true,
      reason: "Receipt date and merchant type matched bundled heuristics.",
      size: 77824,
      modified_at: "2026-07-25T11:35:00Z",
      metadata: { category: "receipt", amount: "36.40 EUR", pages: 1 },
    },
    {
      id: "brief",
      current_name: "client-brief-v2.pdf",
      source_path: `${source}/client-brief-v2.pdf`,
      proposed_name: "20260724-brief-website-refresh-v2.pdf",
      status: "ready",
      included: true,
      reason: "The existing version and document title supplied the naming fields.",
      size: 236544,
      modified_at: "2026-07-24T17:05:00Z",
      metadata: { category: "brief", version: "v2", pages: 5 },
    },
    {
      id: "named",
      current_name: "20260720-report-quarterly-review.pdf",
      source_path: `${source}/20260720-report-quarterly-review.pdf`,
      proposed_name: null,
      status: "skipped",
      included: false,
      reason: "Skipped because the filename already matches the configured pattern.",
      size: 503808,
      modified_at: "2026-07-20T10:00:00Z",
      metadata: { category: "report", pages: 12 },
    },
  ],
};

const selectedIds = new Set(plan.items.filter((item) => item.included).map((item) => item.id));

function report(): Report {
  const items = plan.items
    .filter((item) => selectedIds.has(item.id))
    .map((item) => ({
      item_id: item.id,
      source_name: item.current_name,
      target_name: item.proposed_name,
      status: "renamed" as const,
      reason: "Simulated outcome. No file was changed.",
    }));
  return {
    id: reportId,
    plan_id: planId,
    source,
    started_at: "2026-07-29T09:01:00Z",
    completed_at: "2026-07-29T09:01:01Z",
    items,
    counts: { renamed: items.length, skipped: 0, unchanged: 0, failed: 0, cancelled: 0 },
  };
}

function completedRun(kind: "preview" | "apply"): Run {
  return {
    id: `demo-${kind}-run`,
    kind,
    state: "completed",
    completed: kind === "preview" ? plan.items.length : selectedIds.size,
    total: kind === "preview" ? plan.items.length : selectedIds.size,
    current_file: "",
    message: "Simulation complete. No local files were accessed.",
    plan_id: kind === "preview" ? planId : null,
    report_id: kind === "apply" ? reportId : null,
    error: null,
  };
}

export const api: typeof import("./api").api = {
  bootstrap: () => Promise.resolve(bootstrap),
  filesystem: (path: string): Promise<DirectoryListing> =>
    Promise.resolve({
      path: path.startsWith("/demo") ? path : source,
      parent: null,
      entries: [],
      pdf_count: 6,
    }),
  startPreview: () => Promise.resolve({ run_id: "demo-preview-run" }),
  run: (id: string) => Promise.resolve(completedRun(id.includes("apply") ? "apply" : "preview")),
  cancel: (id: string) => Promise.resolve(completedRun(id.includes("apply") ? "apply" : "preview")),
  plan: () => Promise.resolve(plan),
  apply: (_id: string, _revision: number, ids: string[]) => {
    selectedIds.clear();
    ids.forEach((id) => selectedIds.add(id));
    return Promise.resolve({ run_id: "demo-apply-run" });
  },
  report: () => Promise.resolve(report()),
};
