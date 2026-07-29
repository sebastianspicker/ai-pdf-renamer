import { type MouseEvent, type ReactNode } from "react";
import { FolderIcon } from "../icons";
import { compactPath } from "./format";
import { PrivacyChip } from "./PrivacyChip";
import { StageSpine } from "./StageSpine";

type AppChromeProps = {
  active: 1 | 2 | 3;
  source?: string;
  sourceMeta?: string;
  external: boolean;
  children: ReactNode;
  footer?: ReactNode;
  overlays?: ReactNode;
};

function navigateHome(event: MouseEvent<HTMLAnchorElement>) {
  event.preventDefault();
  window.history.pushState({}, "", "/");
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function AppHeader({ source, sourceMeta, external }: Pick<AppChromeProps, "source" | "sourceMeta" | "external">) {
  return (
    <header className="instrument-bar">
      <div className="brand-block">
        <a aria-label="Folionym" className="brand" href="/" onClick={navigateHome}>
          <span aria-hidden="true" className="brand-mark">
            <svg fill="none" viewBox="0 0 24 24">
              <rect height="18" rx="1.5" stroke="currentColor" strokeWidth="1.6" width="16" x="4" y="3" />
              <path d="M8 8h8M8 12h8M8 16h5" stroke="currentColor" strokeLinecap="round" strokeWidth="1.6" />
            </svg>
          </span>
          <span className="brand-word">Folionym</span>
        </a>
        <span className="brand-meta">Local rename instrument</span>
      </div>

      {source ? (
        <div className="scope-chip" title={source}>
          <span className="scope-label">Scope</span>
          <span className="scope-path">
            <FolderIcon size={15} />
            {compactPath(source)}
          </span>
          {sourceMeta ? <span className="scope-count">{sourceMeta}</span> : null}
        </div>
      ) : (
        <div />
      )}

      <PrivacyChip external={external} />
    </header>
  );
}

export function AppChrome({
  active,
  source,
  sourceMeta,
  external,
  children,
  footer,
  overlays,
}: AppChromeProps) {
  // Preview fills Folio columns (filter-rail · ledger · inspector).
  // Source / Apply use a single content surface beside the stage spine.
  const bodyClass = active === 2 ? "body-grid body-grid--preview" : "body-grid body-grid--simple";

  return (
    <div className="app" data-theme="folio">
      <AppHeader external={external} source={source} sourceMeta={sourceMeta} />

      <div className={bodyClass}>
        <StageSpine active={active} />
        {active === 2 ? children : <div className="app-content">{children}</div>}
      </div>

      {footer ? <footer className="consequence">{footer}</footer> : null}
      {overlays}
    </div>
  );
}
