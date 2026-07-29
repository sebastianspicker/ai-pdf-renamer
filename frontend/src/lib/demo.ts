export const isStaticDemo = import.meta.env.VITE_FOLIONYM_DEMO === "true";

export function demoAsset(path: string): string {
  return `${import.meta.env.BASE_URL}${path.replace(/^\//, "")}`;
}
