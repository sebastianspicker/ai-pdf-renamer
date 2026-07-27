# Security Policy

## Supported versions

| Version | Support |
| --- | --- |
| Current `0.4.0` alpha line | Security fixes during active alpha development |
| Earlier development milestones | Not supported |

Alpha compatibility is intentionally limited. If a prerelease exists, upgrade
to the newest published prerelease before reporting a defect unless the issue
prevents that upgrade. Otherwise, include the candidate version or commit you
tested.

## Local LLM traffic

The built-in HTTP-only LLM client accepts structurally valid HTTP(S) endpoint
URLs. It rejects missing or malformed authorities, embedded credentials,
unsupported schemes, unsafe characters, and invalid ports or hostnames before
creating a request. Text, chat, vision, and diagnostic requests do not follow
redirects.

In the main CLI path, the default is the preset-driven local Ollama endpoint
`http://127.0.0.1:11434/v1/completions`; the HTTP client falls back to
`http://127.0.0.1:8080/v1/completions` when no preset or explicit override is
applied. The client uses `trust_env=False` so proxy environment variables do
not reroute PDF-derived model request content.

Custom HTTP clients or wrapper scripts may inherit proxy settings. Set
`NO_PROXY=127.0.0.1,localhost` when required, and route local LLM traffic
through a proxy only when that proxy is allowed to inspect PDF-derived content.

LLM error logs omit endpoint URLs, request exception details, response bodies,
and document-derived payloads. Some OpenAI-compatible servers echo request
content in error responses, so response bodies must not be written to logs.

The release does not support local model-loading or embedding backends. Keep
document-derived model request content within an endpoint and network boundary you
explicitly trust.

For a non-loopback LLM endpoint, use HTTPS to protect PDF content in transit.
Plain HTTP to a remote host transmits document text unencrypted. The default
policy logs a warning; `--require-https` or `FOLIONYM_REQUIRE_HTTPS=1` rejects
the configuration. Literal loopback addresses, including `127.0.0.0/8` and
`::1`, and the exact hostname `localhost` may use HTTP.

For a non-loopback post-rename hook URL, use HTTPS to protect the metadata
payload in transit. Hook HTTP policy is stricter and accepts plain HTTP only
for literal loopback IP addresses.

## Local browser frontend

`folionym-web` binds only to `127.0.0.1`. Its API requires a process-local,
HTTP-only same-site cookie, a recognized loopback Host header, same-origin
mutations, and JSON request bodies. Responses set a restrictive Content
Security Policy and disable framing. The folder navigator returns directory
names and direct PDF counts to the local browser, not PDF contents.

The browser frontend does not make a configured non-loopback model endpoint
local. Before the first Preview for each exact external endpoint, it requires
the operator to acknowledge that document-derived content may leave the
machine.

## Input and resource boundaries

Folionym processes PDFs and invokes optional native PDF and OCR tooling with the
current user's operating-system permissions. It does not sandbox parsers or OCR
processes. Use documents from a trusted source or an isolated account when the
input may be hostile.

The alpha has no hard input-byte limit, no maximum worker count, and no
application-level OCR timeout. Page extraction is unlimited unless
`--max-pages-for-extraction` is set. Large, malformed, or adversarial files can
consume substantial CPU, memory, disk space, or process time. Run one Folionym
process per target directory and apply explicit limits for untrusted workloads.

## Logs and local caches

Treat log files, persistent LLM cache files, metadata exports, summary JSON, and rename logs as document-adjacent data. They can include filenames, categories, summaries, keywords, paths, or model responses derived from private PDFs.

Persistent LLM cache values remain plaintext local JSON. The tool creates cache directories and files with owner-only permissions where the platform supports POSIX-style permissions, but backups, sync tools, or a custom cache location can still copy the data elsewhere.

Use `--no-cache` for sensitive one-off runs, or keep `--cache-dir` on a private local filesystem. Avoid `--explain` on sensitive documents unless you intend to keep detailed classification reasoning, including document-derived summaries or keywords, in the configured log sink.

## Post-rename hook

The optional post-rename hook (`FOLIONYM_POST_RENAME_HOOK` or config) supports HTTP(S) endpoints only. Local command hooks are refused and logged as warnings. Old path, new path, and metadata are sent as JSON fields:

- `old_path`
- `new_path`
- `meta`

Plain HTTP is allowed only for literal loopback IPv4 or IPv6 endpoints, such as
`http://127.0.0.1:8000/hook` or `http://[::1]:8000/hook`. Hostnames such as
`localhost` are not accepted for plain HTTP because DNS resolution is outside
the validation boundary. Use HTTPS for non-loopback hook receivers. Malformed
URLs, URLs with embedded credentials, and local commands are rejected before a
network session is created.

Send hook payloads only to receivers you operate or have explicitly approved for these document paths and metadata. The payload includes document-adjacent paths and metadata.
Hook requests do not follow redirects, so an approved endpoint cannot redirect
the payload to a URL outside this transport policy.

Hook failures are non-fatal. Their log records intentionally omit the endpoint
URL, request payload, document paths, metadata, and response body.

## Reporting a vulnerability

If you discover a security vulnerability, please avoid creating a public issue.
Instead, use the repository's
[private security-advisory form](https://github.com/sebastianspicker/AI-PDF-Renamer/security/advisories/new)
when it is available to your GitHub account.

If private advisories are unavailable, do not put vulnerability details,
document samples, paths, logs, or reproduction payloads in a public issue.
Open a minimal issue asking the maintainer to arrange a private reporting
channel, then wait for that channel before sharing technical details.
