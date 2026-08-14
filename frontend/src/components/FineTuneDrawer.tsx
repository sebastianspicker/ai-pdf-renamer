import type { ChangeEvent, ReactNode } from "react";
import type { Settings, SettingsBooleanKey, SettingsTextKey } from "../types";
import { Button } from "./Button";
import { Field } from "./Field";
import { Switch } from "./Switch";
import { TextInput } from "./TextInput";

type SettingsChangeHandler = (key: keyof Settings, value: Settings[keyof Settings]) => void;

function createTextSettingChangeHandler(setting: SettingsTextKey, onChange: SettingsChangeHandler) {
  return (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    onChange(setting, event.target.value);
  };
}

type TextSettingProps = {
  hint?: string;
  inputMode?: "numeric";
  label: string;
  onChange: SettingsChangeHandler;
  placeholder?: string;
  setting: SettingsTextKey;
  settings: Settings;
};

const OUTPUT_TEXT_SETTINGS = [
  { label: "Backup folder", setting: "backup_dir" },
  { label: "Rename log", setting: "rename_log" },
  { label: "Metadata export", setting: "export_metadata" },
  { label: "Summary JSON", setting: "summary_json" },
  { label: "Rules file", setting: "rules_file" },
] as const satisfies ReadonlyArray<Pick<TextSettingProps, "label" | "setting">>;

function TextSetting({ hint, inputMode, label, onChange, placeholder, setting, settings }: TextSettingProps) {
  return (
    <Field hint={hint} label={label}>
      <TextInput
        inputMode={inputMode}
        onChange={createTextSettingChangeHandler(setting, onChange)}
        placeholder={placeholder}
        value={settings[setting]}
      />
    </Field>
  );
}

function SelectSetting({
  children,
  label,
  onChange,
  setting,
  settings,
}: TextSettingProps & { children: ReactNode }) {
  return (
    <Field label={label}>
      <select
        className="select"
        onChange={createTextSettingChangeHandler(setting, onChange)}
        value={settings[setting]}
      >
        {children}
      </select>
    </Field>
  );
}

type ToggleSettingProps = {
  description?: string;
  label: string;
  onChange: SettingsChangeHandler;
  setting: SettingsBooleanKey;
  settings: Settings;
};

const PROCESSING_TOGGLES = [
  { label: "OCR scanned PDFs", setting: "use_ocr" },
  { label: "Vision fallback", setting: "use_vision_fallback" },
  { label: "Vision first", setting: "vision_first" },
  { label: "Use structured fields", setting: "use_structured_fields" },
  { label: "Use PDF metadata date", setting: "use_pdf_metadata_date" },
  { label: "Skip already named PDFs", setting: "skip_already_named" },
  { label: "Include subfolders", setting: "recursive" },
] as const satisfies ReadonlyArray<Pick<ToggleSettingProps, "label" | "setting">>;

function ToggleSetting({
  description,
  label,
  onChange,
  setting,
  settings,
}: ToggleSettingProps) {
  return (
    <Switch
      checked={settings[setting]}
      description={description}
      label={label}
      onChange={(value) => {
        onChange(setting, value);
      }}
    />
  );
}

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
  onChange: SettingsChangeHandler;
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
              <SelectSetting label="Language" onChange={onChange} setting="language" settings={settings}>
                <option value="de">German</option>
                <option value="en">English</option>
                <option value="fr">French</option>
                <option value="es">Spanish</option>
              </SelectSetting>
              <SelectSetting label="Case" onChange={onChange} setting="case" settings={settings}>
                <option value="kebabCase">kebab-case</option>
                <option value="snake_case">snake_case</option>
                <option value="camelCase">camelCase</option>
                <option value="Title Case">Title Case</option>
              </SelectSetting>
              <SelectSetting label="Date order" onChange={onChange} setting="date_format" settings={settings}>
                <option value="dmy">Day / month / year</option>
                <option value="ymd">Year / month / day</option>
                <option value="mdy">Month / day / year</option>
              </SelectSetting>
              <SelectSetting label="Preset" onChange={onChange} setting="preset" settings={settings}>
                <option value="">Default</option>
                <option value="fast">Fast</option>
                <option value="scanned">Scanned documents</option>
                <option value="high-confidence-heuristic">High-confidence heuristic</option>
              </SelectSetting>
            </div>
            <TextSetting
              hint="Optional. Overrides the standard naming structure."
              label="Filename template"
              onChange={onChange}
              placeholder="{date}-{category}-{subject}"
              setting="template"
              settings={settings}
            />
            <div className="field-grid">
              <TextSetting label="Project" onChange={onChange} setting="project" settings={settings} />
              <TextSetting label="Version" onChange={onChange} setting="version" settings={settings} />
            </div>
          </SettingsSection>
          <SettingsSection title="Processing">
            <ToggleSetting
              description="Use the configured compatible endpoint for enrichment."
              label="Model assistance"
              onChange={onChange}
              setting="use_llm"
              settings={settings}
            />
            {settings.use_llm && (
              <>
                <TextSetting
                  label="Model endpoint"
                  onChange={onChange}
                  placeholder="http://127.0.0.1:11434/v1/completions"
                  setting="llm_url"
                  settings={settings}
                />
                <div className="field-grid">
                  <TextSetting label="Model" onChange={onChange} setting="llm_model" settings={settings} />
                  <TextSetting
                    inputMode="numeric"
                    label="Workers"
                    onChange={onChange}
                    setting="workers"
                    settings={settings}
                  />
                </div>
              </>
            )}
            {PROCESSING_TOGGLES.map((toggle) => (
              <ToggleSetting key={toggle.setting} {...toggle} onChange={onChange} settings={settings} />
            ))}
          </SettingsSection>
          <SettingsSection title="Output">
            {OUTPUT_TEXT_SETTINGS.map((field) => (
              <TextSetting key={field.setting} {...field} onChange={onChange} settings={settings} />
            ))}
            <ToggleSetting
              label="Write PDF metadata"
              onChange={onChange}
              setting="write_pdf_metadata"
              settings={settings}
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
