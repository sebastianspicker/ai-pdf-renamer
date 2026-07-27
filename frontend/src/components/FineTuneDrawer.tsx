import type { ReactNode } from "react";
import type { Settings } from "../types";
import { Button } from "./Button";
import { Field } from "./Field";
import { Switch } from "./Switch";
import { TextInput } from "./TextInput";

export function SettingsSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="settings-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export function FineTuneDrawer({
  open,
  settings,
  onChange,
  onClose,
}: {
  open: boolean;
  settings: Settings;
  onChange: (key: keyof Settings, value: Settings[keyof Settings]) => void;
  onClose: () => void;
}) {
  if (!open) return null;
  return (
    <>
      <button aria-label="Close fine-tune settings" className="drawer-scrim" onClick={onClose} />
      <aside aria-label="Fine-tune settings" className="drawer drawer--open">
        <header className="drawer__header">
          <div>
            <span className="eyebrow">Settings</span>
            <h2>Fine-tune</h2>
          </div>
          <button aria-label="Close fine-tune settings" className="icon-button" onClick={onClose}>
            ×
          </button>
        </header>
        <div className="drawer__body">
          <SettingsSection title="Naming">
            <div className="field-grid">
              <Field label="Language">
                <select
                  className="select"
                  onChange={(event) => onChange("language", event.target.value)}
                  value={settings.language}
                >
                  <option value="de">German</option>
                  <option value="en">English</option>
                  <option value="fr">French</option>
                  <option value="es">Spanish</option>
                </select>
              </Field>
              <Field label="Case">
                <select className="select" onChange={(event) => onChange("case", event.target.value)} value={settings.case}>
                  <option value="kebabCase">kebab-case</option>
                  <option value="snake_case">snake_case</option>
                  <option value="camelCase">camelCase</option>
                  <option value="Title Case">Title Case</option>
                </select>
              </Field>
              <Field label="Date order">
                <select
                  className="select"
                  onChange={(event) => onChange("date_format", event.target.value)}
                  value={settings.date_format}
                >
                  <option value="dmy">Day / month / year</option>
                  <option value="ymd">Year / month / day</option>
                  <option value="mdy">Month / day / year</option>
                </select>
              </Field>
              <Field label="Preset">
                <select
                  className="select"
                  onChange={(event) => onChange("preset", event.target.value)}
                  value={settings.preset}
                >
                  <option value="">Default</option>
                  <option value="fast">Fast</option>
                  <option value="scanned">Scanned documents</option>
                  <option value="high-confidence-heuristic">High-confidence heuristic</option>
                </select>
              </Field>
            </div>
            <Field hint="Optional. Overrides the standard naming structure." label="Filename template">
              <TextInput
                onChange={(event) => onChange("template", event.target.value)}
                placeholder="{date}-{category}-{subject}"
                value={settings.template}
              />
            </Field>
            <div className="field-grid">
              <Field label="Project">
                <TextInput onChange={(event) => onChange("project", event.target.value)} value={settings.project} />
              </Field>
              <Field label="Version">
                <TextInput onChange={(event) => onChange("version", event.target.value)} value={settings.version} />
              </Field>
            </div>
          </SettingsSection>
          <SettingsSection title="Processing">
            <Switch
              checked={settings.use_llm}
              description="Use the configured compatible endpoint for enrichment."
              label="Model assistance"
              onChange={(value) => onChange("use_llm", value)}
            />
            {settings.use_llm && (
              <>
                <Field label="Model endpoint">
                  <TextInput
                    onChange={(event) => onChange("llm_url", event.target.value)}
                    placeholder="http://127.0.0.1:11434/v1/completions"
                    value={settings.llm_url}
                  />
                </Field>
                <div className="field-grid">
                  <Field label="Model">
                    <TextInput
                      onChange={(event) => onChange("llm_model", event.target.value)}
                      value={settings.llm_model}
                    />
                  </Field>
                  <Field label="Workers">
                    <TextInput
                      inputMode="numeric"
                      onChange={(event) => onChange("workers", event.target.value)}
                      value={settings.workers}
                    />
                  </Field>
                </div>
              </>
            )}
            <Switch checked={settings.use_ocr} label="OCR scanned PDFs" onChange={(value) => onChange("use_ocr", value)} />
            <Switch
              checked={settings.use_vision_fallback}
              label="Vision fallback"
              onChange={(value) => onChange("use_vision_fallback", value)}
            />
            <Switch checked={settings.vision_first} label="Vision first" onChange={(value) => onChange("vision_first", value)} />
            <Switch
              checked={settings.use_structured_fields}
              label="Use structured fields"
              onChange={(value) => onChange("use_structured_fields", value)}
            />
            <Switch
              checked={settings.use_pdf_metadata_date}
              label="Use PDF metadata date"
              onChange={(value) => onChange("use_pdf_metadata_date", value)}
            />
            <Switch
              checked={settings.skip_already_named}
              label="Skip already named PDFs"
              onChange={(value) => onChange("skip_already_named", value)}
            />
            <Switch checked={settings.recursive} label="Include subfolders" onChange={(value) => onChange("recursive", value)} />
          </SettingsSection>
          <SettingsSection title="Output">
            <Field label="Backup folder">
              <TextInput onChange={(event) => onChange("backup_dir", event.target.value)} value={settings.backup_dir} />
            </Field>
            <Field label="Rename log">
              <TextInput onChange={(event) => onChange("rename_log", event.target.value)} value={settings.rename_log} />
            </Field>
            <Field label="Metadata export">
              <TextInput
                onChange={(event) => onChange("export_metadata", event.target.value)}
                value={settings.export_metadata}
              />
            </Field>
            <Field label="Summary JSON">
              <TextInput onChange={(event) => onChange("summary_json", event.target.value)} value={settings.summary_json} />
            </Field>
            <Field label="Rules file">
              <TextInput onChange={(event) => onChange("rules_file", event.target.value)} value={settings.rules_file} />
            </Field>
            <Switch
              checked={settings.write_pdf_metadata}
              label="Write PDF metadata"
              onChange={(value) => onChange("write_pdf_metadata", value)}
            />
          </SettingsSection>
        </div>
        <footer className="drawer__footer">
          <Button onClick={onClose} variant="primary">
            Done
          </Button>
        </footer>
      </aside>
    </>
  );
}
