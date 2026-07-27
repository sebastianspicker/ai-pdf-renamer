/** True when model assistance is off or the configured endpoint is loopback. */
export function isLocalEndpoint(settings: {
  use_llm: boolean;
  llm_url: string;
}): boolean {
  return (
    !settings.use_llm ||
    !settings.llm_url ||
    settings.llm_url.includes("127.0.0.1") ||
    settings.llm_url.includes("localhost")
  );
}
