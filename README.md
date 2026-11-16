# backend-switchboard

One local endpoint that lets Claude Code switch between model backends with `/model`, in a single session.

| `/model` value | Goes to |
|---|---|
| `claude-sonnet-5-5` (no tag) | Anthropic, using your own Claude Code login (forwarded untouched) |
| `claude-sonnet-5-5@argo` | [Argo](https://github.com/n-getty/argo-shim) via a running `argo-shim` |
| `claude-opus-5-5@google@mag` | AMSC MAG (LiteLLM-style, Bearer auth); `@mag` is stripped, `@google` is MAG's own suffix |
| `beam@alcf`, `muse@alcf` | ALCF inference endpoints (native Anthropic Messages API) |
| `google-claude-opus-5@asksage` | AskSage, translated in-process to `/query` |
| `claude-opus-5-5@copilot`, `gpt-6.1-sol@copilot` | GitHub Copilot via your `gh` login |

Claude Code accepts only one `ANTHROPIC_BASE_URL`, so the switch happens behind it: the router reads the model name,
picks a backend, swaps in that backend's credentials and streams the response back. Standard library only, Python 3.8+.

> Not affiliated with Anthropic. The Anthropic route forwards your Claude Code login unchanged and never reads, stores
> or logs it. Check your plan's terms before sending subscription traffic through a local proxy.

## Quick start

```bash
pip install backend-switchboard        # or: uvx backend-switchboard run
backend-switchboard init               # creates the local key and Claude Code settings
backend-switchboard doctor             # checks each backend's setup, flags settings that break your subscription
backend-switchboard run                # starts the router if needed, then launches claude (extra args go to claude)
```

Then in Claude Code: `/model beam@alcf`, `/model claude-sonnet-5-5@argo`, `/model claude-sonnet-5-5` (back to your
subscription). Switching mid-conversation carries the history across backends. Run `/login` once in your normal
`~/.claude`; no per-backend config directory or alias is needed.

Other commands: `migrate`, `ensure` (start in background), `stop`, `serve` (foreground). State lives in
`~/.backend-switchboard/` (override with `BACKEND_SWITCHBOARD_HOME`); the log is `router.log` there.

## Coming from an Argo-only setup

If `~/.claude/settings.json` already points at argo-shim (`ANTHROPIC_BASE_URL`, `apiKeyHelper`,
`CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1`), Claude Code skips your login and sends the shim key to Anthropic. `run` warns
when it sees this. Fix it once:

1. **Close any running Argo Claude Code sessions.** Claude Code re-reads `settings.json` live, so a running session
   would fall back to your Anthropic login and fail with "Login expired".
2. `backend-switchboard migrate`. It saves the shim URL and token to `config.json` (mode 600), backs up
   `settings.json` (mode 600, contains the token; delete when happy) and removes the three settings, leaving the rest.
   If `CLAUDE_CODE_SKIP_ANTHROPIC_AUTH` is exported in your shell profile, it tells you what to unset.
3. `backend-switchboard doctor`, `backend-switchboard run`, then `/login` once.

By hand: put `{"argo_url": "http://127.0.0.1:<port>/argoapi", "argo_token": "<shim token>"}` in `config.json` (or export
`ARGO_SHIM_URL` / `ARGO_SHIM_TOKEN` where the router starts) and delete those three settings.

## Backend setup

Export credentials in the environment the router starts from, and `backend-switchboard stop` to pick up changes.

- **Anthropic (no tag):** nothing to configure.
- **Argo (`@argo`):** run argo-shim as usual. The router re-reads its token and URL from `~/.claude/settings.json` (or
  `config.json`) on each request. An "ACCESS REVOKED" 200 from an exhausted quota is turned into a 429.
- **MAG (`@mag`):** export `AMSC_TOKEN`. Set `mag_url` to use another Bearer-auth gateway.
- **ALCF (`@alcf`):** `uv tool install alcf-ai alcf-tokens`, then `alcf-tokens login inference` once. The router runs
  `alcf-tokens get-token inference` (cached 2 min) or uses `ALCF_INFERENCE_TOKEN`. `beam` and `muse` are aliases for
  `beam-1.0-fp8` (on the `tara` cluster) and `muse-spark-bf16`; other names pass through. Add aliases under
  `aliases.alcf`: a model name, or `{"model": ..., "url": ...}` for another cluster.
- **AskSage (`@asksage`):** export `ASKSAGE_API_KEY` (from your AskSage account settings; `ASKSAGE_PROXY_KEY` also
  works). For a private CA, set `asksage_cert` in `config.json`; `asksage_url` defaults to
  `https://api.asksage.anl.gov/server`. Use AskSage's model ids (`google-claude-opus-5@asksage`); list yours by POSTing
  to `<asksage_url>/get-models` with an `x-access-tokens` header. Tool calls and results are written into history as
  prose and streaming is emulated; Claude models handle the tool loop, weaker ones (`gpt-4.1`) often stall.
- **GitHub Copilot (`@copilot`):** `gh auth login` (needs a Copilot seat), or set `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`
  or `GITHUB_TOKEN`. Claude models are forwarded natively (dashed names map to dotted: `claude-opus-5-5` becomes
  `claude-opus-5.5`); `gpt-*`, `grok-*`, `mai-*` are translated to `/responses` with emulated streaming; `gemini-*`
  is rejected (400). Add short names under `aliases.copilot`.

A bare `name@google` is rejected as ambiguous; always add a router tag.

**TLS errors on every HTTPS backend** (`SSLError ... (_ssl.c:3036)`): Python rejects the system `openssl.cnf`. The
router restarts itself with `OPENSSL_CONF=/dev/null`; `doctor` reports it.

## Tool search

`init` sets `ENABLE_TOOL_SEARCH=true`, which defers tool definitions to shrink requests. For every non-Anthropic route
the router drops the `defer_loading` placeholder and rewrites `tool_reference` blocks into short text results. To send
all tools every time, set `"tool_search": "false"` in `config.json` (`auto:N` also works).

## Configuration

Optional `~/.backend-switchboard/config.json`; every key has a default.

| key | default | meaning |
|---|---|---|
| `port` | `8790` | local listen port (loopback only) |
| `tool_search` | `true` | value written to `ENABLE_TOOL_SEARCH` |
| `anthropic_url` | `https://api.anthropic.com` | no-tag route |
| `mag_url` | AMSC MAG host | `@mag` route |
| `argo_url` / `argo_token` | `$ARGO_SHIM_URL` / `$ARGO_SHIM_TOKEN`, else from `argo_settings` | argo-shim URL (include `/argoapi`) and token |
| `argo_settings` | `~/.claude/settings.json` | fallback file with `apiKeyHelper` and `env.ANTHROPIC_BASE_URL` |
| `asksage_url` / `asksage_cert` | `https://api.asksage.anl.gov/server` / none | AskSage server, optional CA bundle |
| `copilot_url` | `https://api.githubcopilot.com` | `@copilot` route |
| `alcf_url` | ALCF Minerva `/api` | `@alcf` route |
| `models` | one row per backend; `[]` for none | extra `/model` picker rows (`"beam@alcf"` or `{"model", "label", "description"}`), written as `modelPicker` (Claude Code 2.1.242+) |
| `aliases` | `beam`, `muse` | short names per backend (`alcf`, `copilot`) |

## Security

- Listens on `127.0.0.1` only and requires a random key on every request (sent via `ANTHROPIC_CUSTOM_HEADERS`), so other
  users on a shared login node can't spend your credentials. The key and `settings.json` are mode 600.
- Upstream credentials are read per request and never logged; the log has model, backend, status and duration only.

## Known limits

- Argo and MAG are covered by mock-upstream tests only; Anthropic, ALCF and cross-backend switching were tested live.
- If a backend is down or returns 5xx, Claude Code retries for minutes, so a stopped Argo tunnel looks like a hang.
  Check `router.log`.
- ALCF has no token-count endpoint (404); Claude Code falls back. Muse is inconsistent at tool loops (model behavior).
- Switching Anthropic <-> ALCF works, except with extended thinking enabled on the Anthropic side.
- Not covered: the interactive `/model` picker UI (typing the name works) and cross-backend prompt-cache behavior.

## Development

`python3 tests/selftest.py` (see [tests/README.md](tests/README.md)). Merging to `main` bumps the version, updates the
changelog and publishes to PyPI.

## License

MIT
