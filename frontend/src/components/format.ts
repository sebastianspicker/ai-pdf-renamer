export function compactPath(path: string): string {
  const homeName = path.split("/").filter(Boolean);
  if (homeName.length <= 2) return path;
  return `…/${homeName.slice(-2).join("/")}`;
}

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
