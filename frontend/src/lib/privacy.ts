const HTTP_ENDPOINT_RE = /^https?:\/\/(\[[^\]]+\]|[^:/?#]+)(?::\d+)?(?:[/?#]|$)/i;
const UNSAFE_URL_CHARACTER_RE = /[\u0000-\u0020\u007f\\]/;
const HTTP_PROTOCOLS = new Set(["http:", "https:"]);

function parseIpv6Groups(hostname: string): number[] | null {
  const halves = hostname.split("::");
  if (halves.length > 2) return null;
  const parseHalf = (value: string) =>
    value ? value.split(":").map((part) => (/^[0-9a-f]{1,4}$/i.test(part) ? Number.parseInt(part, 16) : -1)) : [];
  const left = parseHalf(halves[0]);
  const right = parseHalf(halves[1] ?? "");
  if ([...left, ...right].some((part) => part < 0)) return null;
  if (halves.length === 1) return left.length === 8 ? left : null;
  const omittedGroups = 8 - left.length - right.length;
  return omittedGroups > 0 ? [...left, ...Array<number>(omittedGroups).fill(0), ...right] : null;
}

function isIpv6Loopback(hostname: string): boolean {
  const groups = parseIpv6Groups(hostname);
  if (!groups) return false;
  const nativeLoopback = groups.slice(0, 7).every((group) => group === 0) && groups[7] === 1;
  const mappedIpv4Loopback =
    groups.slice(0, 5).every((group) => group === 0) &&
    groups[5] === 0xffff &&
    groups[6] >> 8 === 127;
  return nativeLoopback || mappedIpv4Loopback;
}

function isIpv4Loopback(hostname: string): boolean {
  const octets = hostname.split(".");
  if (octets.length !== 4) return false;
  const canonical = octets.every((octet) => /^(?:0|[1-9]\d{0,2})$/.test(octet) && Number(octet) <= 255);
  return canonical && Number(octets[0]) === 127;
}

function hasSafeHttpAuthority(endpoint: URL): boolean {
  return HTTP_PROTOCOLS.has(endpoint.protocol) && !endpoint.username && !endpoint.password && !endpoint.hash;
}

function isLoopbackAuthority(rawAuthority: string, parsedHostname: string): boolean {
  const rawHostname = rawAuthority.replace(/^\[|\]$/g, "").toLowerCase();
  if (rawHostname === "localhost" || isIpv4Loopback(rawHostname)) return true;
  return rawHostname.includes(":") && isIpv6Loopback(parsedHostname.replace(/^\[|\]$/g, "").toLowerCase());
}

function isLoopbackUrl(value: string): boolean {
  const normalized = value.trim();
  if (!normalized || UNSAFE_URL_CHARACTER_RE.test(normalized)) return false;
  const authorityMatch = HTTP_ENDPOINT_RE.exec(normalized);
  if (!authorityMatch) return false;
  try {
    const endpoint = new URL(normalized);
    return hasSafeHttpAuthority(endpoint) && isLoopbackAuthority(authorityMatch[1], endpoint.hostname);
  } catch {
    return false;
  }
}

/** True when model assistance is off or the configured endpoint is loopback. */
export function isLocalEndpoint(settings: {
  use_llm: boolean;
  llm_url: string;
}): boolean {
  return !settings.use_llm || !settings.llm_url || isLoopbackUrl(settings.llm_url);
}
