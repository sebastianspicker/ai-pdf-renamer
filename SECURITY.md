# Security Policy

## Local LLM traffic

The built-in LLM client sends requests only to the URL you configure. In the main CLI path, the default is the preset-driven local Ollama endpoint `http://127.0.0.1:11434/v1/completions`; the lower-level HTTP backend still falls back to `http://127.0.0.1:8080/v1/completions` for a plain llama.cpp server when no preset/default override is applied. To avoid routing this traffic through a proxy (e.g. `HTTP_PROXY`), the client uses `trust_env=False` so that PDF-derived prompt content stays on your machine.

**If you use a custom HTTP client or wrapper scripts that inherit proxy settings:** set `NO_PROXY=127.0.0.1,localhost` (or `no_proxy` on some systems) so that requests to the local LLM endpoint are never sent via a proxy. Route local LLM traffic through a proxy only when you explicitly trust that proxy to inspect PDF-derived prompt content.

LLM HTTP error logs intentionally record status and error context without response bodies. Some OpenAI-compatible servers echo request prompts in error responses, so logging response bodies can persist PDF-derived text in local log files.

The in-process backend (`--llm-backend in-process`) loads a GGUF model directly into the process using `llama-cpp-python` and makes no network requests at all.

**If you configure a non-loopback LLM endpoint** (anything other than `127.0.0.1`, `::1`, or `localhost`), use HTTPS (`https://`) to protect PDF content in transit. Plain HTTP to a remote host will transmit document text unencrypted; the tool logs a WARNING in this case but does not block the request.

**If you configure a non-loopback post-rename hook URL**, the same applies: use HTTPS to protect the metadata payload (old path, new path, category, summary) in transit.

## Logs and local caches

Treat log files, persistent LLM cache files, metadata exports, summary JSON, and rename logs as document-adjacent data. They can include filenames, categories, summaries, keywords, paths, or model responses derived from private PDFs.

Persistent LLM cache values remain plaintext local JSON. The tool creates cache directories and files with owner-only permissions where the platform supports POSIX-style permissions, but backups, sync tools, or a custom cache location can still copy the data elsewhere.

Use `--no-cache` for sensitive one-off runs, or keep `--cache-dir` on a private local filesystem. Avoid `--explain` on sensitive documents unless you intend to keep detailed classification reasoning, including document-derived summaries or keywords, in the configured log sink.

## Post-rename hook

The optional post-rename hook (`AI_PDF_RENAMER_POST_RENAME_HOOK` or config) supports HTTP(S) endpoints only. Local command hooks are refused and logged as warnings. Old path, new path, and metadata are sent as JSON fields:

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

## Reporting a Vulnerability

If you discover a security vulnerability, please avoid creating a public issue.
Instead, open a private security advisory on GitHub if available.

If that is not possible, open an issue with minimal details and mark it
as security-related so it can be triaged quickly.
