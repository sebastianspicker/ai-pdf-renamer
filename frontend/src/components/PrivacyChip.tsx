export function PrivacyChip({ external }: { external: boolean }) {
  return (
    <div
      className={`privacy-chip ${external ? "privacy-chip--external" : "privacy-chip--local"}`}
      role="status"
    >
      <span aria-hidden="true" className="privacy-pulse" />
      <div className="privacy-copy">
        {external ? (
          <>
            <strong>External model</strong>
            <span>Document text may leave this machine</span>
          </>
        ) : (
          <>
            <strong>Local only</strong>
            <span>Heuristics · no model call</span>
          </>
        )}
      </div>
    </div>
  );
}
