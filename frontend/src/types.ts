export type RunState = "queued" | "running" | "completed" | "cancelled" | "failed";
export type PreviewStatus = "ready" | "review" | "skipped" | "failed";
export type ApplyStatus = "renamed" | "skipped" | "unchanged" | "failed" | "cancelled";

export type SettingsTextKey =
  | "directory" | "single_file" | "language" | "case"
  | "date_format" | "preset" | "project" | "version"
  | "template" | "backup_dir" | "rename_log" | "export_metadata"
  | "summary_json" | "rules_file" | "post_rename_hook" | "llm_url"
  | "llm_model" | "llm_timeout" | "max_tokens" | "max_content_chars"
  | "max_content_tokens" | "workers" | "max_filename_chars" | "acknowledged_external_endpoint";

export type SettingsBooleanKey =
  | "dry_run" | "use_llm" | "use_ocr"
  | "recursive" | "skip_already_named" | "use_pdf_metadata_date"
  | "use_structured_fields" | "write_pdf_metadata" | "use_vision_fallback"
  | "simple_naming_mode" | "vision_first";

export interface Settings extends Record<SettingsTextKey, string>, Record<SettingsBooleanKey, boolean> {}

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
