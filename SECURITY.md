# Security Policy

## Local LLM traffic

The built-in LLM client sends requests only to the URL you configure (default `http://127.0.0.1:11434`). To avoid routing this traffic through a proxy (e.g. `HTTP_PROXY`), the client uses `trust_env=False` so that PDF-derived prompt content stays on your machine.

**If you use a custom HTTP client or run scripts that might inherit proxy settings:** set `NO_PROXY=127.0.0.1,localhost` (or `no_proxy` on some systems) so that requests to the local LLM endpoint are never sent via a proxy. Otherwise prompt content could leave your machine. See BUGS_AND_FIXES.md §16.

## Post-rename hook

The optional post-rename hook (`AI_PDF_RENAMER_POST_RENAME_HOOK` or config) is executed in a subprocess with **shell=True**. The hook string is **user-defined** and runs with your privileges. Old path, new path, and metadata are passed **only via environment variables** (`AI_PDF_RENAMER_OLD_PATH`, `AI_PDF_RENAMER_NEW_PATH`, `AI_PDF_RENAMER_META`), not interpolated into the hook command.

**Do not** embed PDF content, filenames, or other untrusted input into the hook command string itself (e.g. in config or env). Use the provided env vars inside your script if you need paths or metadata. Keep the hook under your control; if config or env can be influenced by others, they could run arbitrary commands.

## Reporting a Vulnerability

If you discover a security vulnerability, please avoid creating a public issue.
Instead, open a private security advisory on GitHub if available.

If that is not possible, open an issue with minimal details and mark it
as security-related so it can be triaged quickly.
