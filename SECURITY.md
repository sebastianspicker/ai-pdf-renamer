# Security Policy

## Local LLM traffic

The built-in LLM client sends requests only to the URL you configure. In the main CLI path, the default is the preset-driven local Ollama endpoint `http://127.0.0.1:11434/v1/completions`; the lower-level HTTP backend still falls back to `http://127.0.0.1:8080/v1/completions` for a plain llama.cpp server when no preset/default override is applied. To avoid routing this traffic through a proxy (e.g. `HTTP_PROXY`), the client uses `trust_env=False` so that PDF-derived prompt content stays on your machine.

**If you use a custom HTTP client or run scripts that might inherit proxy settings:** set `NO_PROXY=127.0.0.1,localhost` (or `no_proxy` on some systems) so that requests to the local LLM endpoint are never sent via a proxy. Otherwise prompt content could leave your machine.

The in-process backend (`--llm-backend in-process`) loads a GGUF model directly into the process using `llama-cpp-python` and makes no network requests at all.

**If you configure a non-loopback LLM endpoint** (anything other than `127.0.0.1`, `::1`, or `localhost`), use HTTPS (`https://`) to protect PDF content in transit. Plain HTTP to a remote host will transmit document text unencrypted; the tool logs a WARNING in this case but does not block the request.

**If you configure a non-loopback post-rename hook URL**, the same applies: use HTTPS to protect the metadata payload (old path, new path, category, summary) in transit.

## Post-rename hook

The optional post-rename hook (`AI_PDF_RENAMER_POST_RENAME_HOOK` or config) runs in a subprocess with **shell=False**. The hook string is **operator-defined** and runs with your privileges. Old path, new path, and metadata are passed via environment variables:

- `AI_PDF_RENAMER_OLD_PATH`
- `AI_PDF_RENAMER_NEW_PATH`
- `AI_PDF_RENAMER_META`

If shell metacharacters are detected in the configured command string, the tool explicitly invokes your local shell executable as a subprocess argument (`/bin/sh -lc ...` on Unix, `cmd.exe /c ...` on Windows), still using `shell=False` for process creation.

**Do not** embed PDF content, filenames, or other untrusted input into the hook command string itself (in config or env). Use the provided environment variables inside your script when you need paths or metadata. Keep hook configuration under your control; if config/env is attacker-controlled, arbitrary command execution is possible.

## Reporting a Vulnerability

If you discover a security vulnerability, please avoid creating a public issue.
Instead, open a private security advisory on GitHub if available.

If that is not possible, open an issue with minimal details and mark it
as security-related so it can be triaged quickly.
