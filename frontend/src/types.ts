export type RunState = "queued" | "running" | "completed" | "cancelled" | "failed";
export type PreviewStatus = "ready" | "review" | "skipped" | "failed";
export type ApplyStatus = "renamed" | "skipped" | "unchanged" | "failed" | "cancelled";

export interface Settings {
  directory: string;
  single_file: string;
  language: string;
  case: string;
  date_format: string;
  preset: string;
  project: string;
  version: string;
  template: string;
  backup_dir: string;
  rename_log: string;
  export_metadata: string;
  summary_json: string;
  rules_file: string;
  post_rename_hook: string;
  llm_url: string;
  llm_model: string;
  llm_timeout: string;
  max_tokens: string;
  max_content_chars: string;
  max_content_tokens: string;
  workers: string;
  max_filename_chars: string;
  dry_run: boolean;
  use_llm: boolean;
  use_ocr: boolean;
  recursive: boolean;
  skip_already_named: boolean;
  use_pdf_metadata_date: boolean;
  use_structured_fields: boolean;
  write_pdf_metadata: boolean;
  use_vision_fallback: boolean;
  simple_naming_mode: boolean;
  vision_first: boolean;
  acknowledged_external_endpoint: string;
}

export interface DirectoryEntry {
  name: string;
  path: string;
  pdf_count: number;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  entries: DirectoryEntry[];
  pdf_count: number;
}

export interface Bootstrap {
  settings: Settings;
  roots: DirectoryEntry[];
  capabilities: Record<string, boolean>;
}

export interface Run {
  id: string;
  kind: "preview" | "apply";
  state: RunState;
  completed: number;
  total: number;
  current_file: string;
  message: string;
  plan_id: string | null;
  report_id: string | null;
  error: string | null;
}

export interface PreviewItem {
  id: string;
  current_name: string;
  source_path: string;
  proposed_name: string | null;
  status: PreviewStatus;
  included: boolean;
  reason: string | null;
  size: number;
  modified_at: string | null;
  metadata: Record<string, unknown>;
}

export interface Plan {
  id: string;
  revision: number;
  source: string;
  source_kind: "directory" | "file";
  created_at: string;
  items: PreviewItem[];
  counts: Record<PreviewStatus | "all", number>;
}

export interface ApplyItem {
  item_id: string;
  source_name: string;
  target_name: string | null;
  status: ApplyStatus;
  reason: string | null;
}

export interface Report {
  id: string;
  plan_id: string;
  source: string;
  started_at: string;
  completed_at: string;
  items: ApplyItem[];
  counts: Record<ApplyStatus, number>;
}
