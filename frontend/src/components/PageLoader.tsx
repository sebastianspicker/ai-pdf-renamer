export function PageLoader({ label = "Loading workspace" }: { label?: string }) {
  return (
    <div className="page-state" role="status">
      <span className="spinner" />
      <p>{label}</p>
    </div>
  );
}
