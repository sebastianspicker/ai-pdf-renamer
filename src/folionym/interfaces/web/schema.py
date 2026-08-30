"""Typed HTTP payloads for Folionym's local browser interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UISettingsPayload(BaseModel):
    """Interactive settings shared by the browser and Textual frontends."""

    model_config = ConfigDict(extra="forbid")

    directory: str = ""
    single_file: str = ""
    language: Literal["de", "en"] = "de"
    case: Literal["camelCase", "kebabCase", "snakeCase"] = "kebabCase"
    date_format: Literal["dmy", "mdy"] = "dmy"
    preset: str = ""
    project: str = ""
    version: str = ""
    template: str = ""
    backup_dir: str = ""
    rename_log: str = ""
    export_metadata: str = ""
    summary_json: str = ""
    rules_file: str = ""
    post_rename_hook: str = ""
    llm_url: str = ""
    llm_model: str = ""
    llm_timeout: str = ""
    max_tokens: str = ""
    max_content_chars: str = ""
    max_content_tokens: str = ""
    workers: str = "1"
    max_filename_chars: str = ""
    dry_run: bool = True
    use_llm: bool = True
    use_ocr: bool = False
    recursive: bool = False
    skip_already_named: bool = False
    use_pdf_metadata_date: bool = True
    use_structured_fields: bool = True
    write_pdf_metadata: bool = False
    use_vision_fallback: bool = False
    simple_naming_mode: bool = False
    vision_first: bool = False
    acknowledged_external_endpoint: str = ""


class PreviewRequest(BaseModel):
    """Start one folder or single-PDF preview."""

    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["directory", "file"]
    path: str = Field(min_length=1)
    settings: UISettingsPayload
    acknowledge_external_endpoint: bool = False


class ApplyRequest(BaseModel):
    """Apply a selected subset of an immutable preview plan."""

    model_config = ConfigDict(extra="forbid")

    plan_revision: int = Field(ge=1)
    selected_ids: list[str]


class RunStartedResponse(BaseModel):
    """Identifier returned after a background operation starts."""

    run_id: str


class DirectoryEntry(BaseModel):
    """One navigable local directory."""

    name: str
    path: str
    pdf_count: int = 0


class DirectoryListing(BaseModel):
    """Directory navigator state."""

    path: str
    parent: str | None
    entries: list[DirectoryEntry]
    pdf_count: int
