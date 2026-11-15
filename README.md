# backend-switchboard

One local endpoint that lets Claude Code switch between several model backends with `/model`, in a single session:

| `/model` value | Goes to |
|---|---|
| `claude-sonnet-5-5` (no tag) | Anthropic, using your own Claude Code login. Requests are forwarded untouched. |
| `claude-sonnet-5-5@argo` | [Argo](https://github.com/n-getty/argo-shim) through a running `argo-shim` |
| `claude-opus-5-5@google@mag` | A LiteLLM-style gateway with Bearer auth (AMSC "MAG"); the tag `@mag` is stripped and `@google` is MAG's own suffix |
| `beam@alcf`, `muse@alcf` | ALCF inference endpoints, which serve the Anthropic Messages API natively |
| `google-claude-opus-5@asksage` | AskSage models, translated in-process to AskSage's `/query` API |
| `claude-opus-5-5@copilot`, `gpt-6.1-sol@copilot` | GitHub Copilot, using your `gh` login: Claude models natively, GPT/Grok models translated to `/responses` |

Claude Code only accepts one `ANTHROPIC_BASE_URL` per session, so the switch has to happen behind that URL.
backend-switchboard is that URL. It reads the model name, picks a backend, swaps in that backend's credentials, and
streams the response back. Standard library only, Python 3.8+. No other services to run.

> Not affiliated with Anthropic. The Anthropic route forwards your Claude Code login (OAuth or API key) unchanged and
> never reads, stores or logs it. Check your plan's terms before sending subscription traffic through a local proxy.

```
Claude Code --> 127.0.0.1:8790 (backend-switchboard) --+--> api.anthropic.com        (no tag)
   /model name[@tag]                                   +--> argo-shim                (@argo)
                                                       +--> MAG                      (@mag)
                                                       +--> ALCF Minerva             (@alcf)
                                                       +--> AskSage                  (@asksage)
                                                       +--> GitHub Copilot           (@copilot)
```

## Install

```bash
pip install backend-switchboard
# or without installing:
uvx backend-switchboard run
```

## Use

```bash
backend-switchboard run              # starts the router if needed, then launches claude
backend-switchboard run -c           # anything after "run" is passed to claude
```

Then inside Claude Code: `/model beam@alcf`, `/model claude-sonnet-5-5@argo`, `/model claude-sonnet-5-5` (back to your
subscription). You can switch mid-conversation; the history is carried across backends.

Other commands: `migrate` (move an Argo-only Claude Code setup behind the switchboard), `doctor` (checks each backend's setup and flags settings that will break your subscription), `init` (create the key and settings), `ensure` (start the router in the background), `stop`, `serve`
(foreground). State lives in `~/.backend-switchboard/` (override with `BACKEND_SWITCHBOARD_HOME`).

You do not need a separate config directory or alias per backend: run `/login` once in your normal `~/.claude` and every
backend works from the same session. (If you already use `CLAUDE_CONFIG_DIR` for a subscription, `run` leaves it alone.)

## Backends

**Anthropic (no tag).** Nothing to configure. Whatever login your Claude Code already has is used.

**Argo (`@argo`).** Run [argo-shim](https://github.com/n-getty/argo-shim) as usual. The router re-reads its token and
local URL from `~/.claude/settings.json` on every request (change the path with `argo_settings`). If Argo answers an
exhausted quota with HTTP 200 and an "ACCESS REVOKED" notice, the router turns that into a 429 so it is not shown as a
model answer.

### Coming from an Argo-only setup

If your Claude Code was already pointed at argo-shim (`ANTHROPIC_BASE_URL`, an `apiKeyHelper`, and
`CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1` in `~/.claude/settings.json`), those settings stop your Anthropic subscription
from working through the switchboard: Claude Code skips your login and sends the shim key to Anthropic. `run` prints a
warning when it sees this. One-time fix, so a single `~/.claude` serves every backend:

**Close any running Claude Code sessions that use the Argo setup before migrating.** Claude Code re-reads
`settings.json` while it runs, so removing `apiKeyHelper` under a live Argo session makes it fall back to your Anthropic
login and fail midstream with "Login expired · Please run /login". Any backup of `settings.json` you make also contains
the shim token; keep it mode 600 (or delete it).

Run `backend-switchboard migrate`. It saves the shim URL and token to `~/.backend-switchboard/config.json` (mode 600),
backs up `settings.json` (mode 600, delete it when happy), and removes `ANTHROPIC_BASE_URL`, `apiKeyHelper` and
`CLAUDE_CODE_SKIP_ANTHROPIC_AUTH` from it, leaving your other settings alone. If `CLAUDE_CODE_SKIP_ANTHROPIC_AUTH` is
exported in your shell profile, `migrate` can't remove that; it tells you what to unset. Then run
`backend-switchboard doctor`, `backend-switchboard run` and `/login` once.

By hand, the same thing: put `{"argo_url": "http://127.0.0.1:<port>/argoapi", "argo_token": "<shim token>"}` in
`config.json` (or export `ARGO_SHIM_URL` and `ARGO_SHIM_TOKEN` where the router starts), then delete those three
settings.

You no longer need a second config directory or a per-backend alias.

**MAG (`@mag`).** Export `AMSC_TOKEN` in the environment the router starts from. Set `mag_url` in `config.json` to
point at another Bearer-auth gateway. MAG's own provider suffixes stay in the name: `claude-opus-5-5@google@mag`.

**ALCF (`@alcf`).** Install the ALCF token tools and log in once, in your own terminal:

```bash
uv tool install alcf-ai alcf-tokens
alcf-tokens login inference
```

The router runs `alcf-tokens get-token inference` (cached for two minutes), or uses `ALCF_INFERENCE_TOKEN` if you set it.
ALCF's Minerva endpoint accepts the Anthropic Messages API directly with a Bearer token, so requests are forwarded as they
are, including streaming, tool calls and the models' thinking blocks. Aliases `beam` and `muse` map to `beam-1.0-fp8`
(on ALCF's `tara` cluster, which has its own URL) and `muse-spark-bf16`; any other name is passed through. Add or change aliases under `aliases.alcf` in `config.json`; a value is a model name, or `{"model": ..., "url": ...}` for a model on another cluster.

**AskSage (`@asksage`).** Setup:

1. Get an AskSage API key from your AskSage account settings (for ANL: https://api.asksage.anl.gov) and export it in the
   shell that starts the router, e.g. in `~/.bashrc`: `export ASKSAGE_API_KEY=...` (`ASKSAGE_PROXY_KEY`, the name asksage-proxy used, also works). The router reads it per request and
   never logs it. It reads only the environment, so restart the router after changing it: `backend-switchboard stop`,
   then launch again.
2. If your AskSage host uses a private CA, save its certificate and point at it in `~/.backend-switchboard/config.json`:
   `{"asksage_cert": "~/path/to/asksage.pem"}`. `asksage_url` defaults to `https://api.asksage.anl.gov/server`; change it
   for another AskSage server.
3. Use AskSage's own model ids: `/model google-claude-opus-5@asksage`, `/model gpt-4.1@asksage`. To list the ids your
   account can use, POST to `<asksage_url>/get-models` with an `x-access-tokens: $ASKSAGE_API_KEY` header.

How it works: the router translates Anthropic requests to AskSage's `/query` itself, with no asksage-proxy or llm-rosetta.
The history goes over as `me`/`gpt` turns and `tools` as a JSON list. AskSage has no field for tool results, so earlier
tool calls and their results are written into the history as prose, and answers arrive whole, so streaming is emulated.
Claude models on AskSage handle the full tool loop (Read, ToolSearch, Bash); weaker ones (`gpt-4.1`) often stall on
deferred tools. Token counts are estimates, and some AskSage models return `model_error` while their upstream is down.

**GitHub Copilot (`@copilot`).** Log in once with `gh auth login` (the account needs a Copilot seat). The router runs
`gh auth token` (cached two minutes), or uses `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` if set, and sends
the Copilot client headers. It asks `/models` which endpoint each model supports (cached ten minutes):

- Claude models serve the Anthropic Messages API, so requests are forwarded as they are, with streaming. Claude Code's
  dashed names are mapped to Copilot's dotted ones: `claude-opus-5-5@copilot` becomes `claude-opus-5.5`.
- `gpt-*`, `grok-*` and `mai-*` models only offer OpenAI's `/responses`. The router translates to it in process, including
  tool calls and tool results. Answers are requested whole and streaming is emulated, and token counts for
  `count_tokens` are estimates. Reasoning content is not carried across turns.
- `gemini-*` models only offer `/chat/completions` and are rejected with a 400.

To list your models, GET `https://api.githubcopilot.com/models` with `Authorization: Bearer $(gh auth token)`,
`User-Agent: GitHubCopilot/1.0.105` and `Copilot-Integration-Id: copilot-developer-cli` headers.
Add short names under `aliases.copilot` in `config.json`.

A bare `name@google` is rejected as ambiguous, because more than one backend uses that suffix. Always add a router tag.

**TLS errors on every HTTPS backend** (`SSLError ... (_ssl.c:3036)`): the Python running the router rejects the system
`openssl.cnf`. The router detects this at start-up and restarts itself with `OPENSSL_CONF=/dev/null`; `doctor` reports it.

## Tool search

`init` sets `ENABLE_TOOL_SEARCH=true`, which defers most tool definitions and cuts the request size (about 37% fewer
tokens in one measurement with roughly 26 tools; your mix will differ). Claude Code decides this once per session, not per model, and the
Anthropic-only `tool_reference` blocks it uses are not understood by other backends. For every route except the plain
Anthropic one, the router therefore drops the `defer_loading` placeholder and rewrites each `tool_reference` into a short
text result; Claude Code already sends the loaded tool's full schema on the next request. To send every
tool definition on every request instead, set `"tool_search": "false"` in `config.json` (Claude Code also accepts
`auto:N`).

## Configuration

Optional `~/.backend-switchboard/config.json`; every key has a default:

| key | default | meaning |
|---|---|---|
| `port` | `8790` | local listen port (loopback only) |
| `tool_search` | `true` | value written to `ENABLE_TOOL_SEARCH` in the generated Claude Code settings |
| `anthropic_url` | `https://api.anthropic.com` | no-tag route |
| `mag_url` | the AMSC MAG host | `@mag` route |
| `argo_url` / `argo_token` | `$ARGO_SHIM_URL` / `$ARGO_SHIM_TOKEN`, else read from `argo_settings` | argo-shim base URL (include `/argoapi`) and client token |
| `argo_settings` | `~/.claude/settings.json` | fallback file with `apiKeyHelper` and `env.ANTHROPIC_BASE_URL` |
| `asksage_url` / `asksage_cert` | `https://api.asksage.anl.gov/server` / none | AskSage server and optional extra CA bundle |
| `copilot_url` | `https://api.githubcopilot.com` | `@copilot` route |
| `alcf_url` | ALCF Minerva `/api` | endpoint behind `@alcf` (Anthropic Messages API) |
| `models` | one row per backend (Argo, MAG, ALCF, AskSage, Copilot); `[]` for none | extra rows for `/model`: names like `"beam@alcf"` or `{"model", "label", "description"}`; written as `modelPicker` in the generated settings (Claude Code 2.1.242+) |
| `aliases` | `beam`, `muse` | short names per backend (`alcf`, `copilot`) |

## Security

- The router listens on `127.0.0.1` only and requires a random key on every request, sent by Claude Code through
  `ANTHROPIC_CUSTOM_HEADERS`. This matters on shared login nodes: without it, any other user on the machine could spend
  your Argo, MAG and ALCF credentials. The key and `settings.json` are created with mode 600.
- Upstream credentials are read per request and are never logged. The log records the model name, backend, status and
  duration only.

## Status and limits

- Tested live: your Anthropic login, ALCF Beam and Muse, and a single conversation switched across backends between turns.
  The Argo and MAG routes are covered by mock-upstream tests only.
- Beam handled loading a deferred tool with `ToolSearch` and then calling it reliably in every run. Muse was inconsistent: in two
  runs, one succeeded and one looped on tool calls until the client timed out. All requests returned 200, so this is model
  behavior, not routing.
- When a backend is down or returns a 5xx, Claude Code retries for a few minutes, so a stopped Argo tunnel looks like a
  hang. Check `~/.backend-switchboard/router.log`.
- ALCF does not implement the token-count endpoint (404). Claude Code falls back and the turn completes.
- Switching between an Anthropic model and ALCF models worked in testing (history includes the ALCF thinking blocks), but not
  with extended thinking enabled on the Anthropic side.
- Not covered: the interactive `/model` picker UI (typing the name works), and prompt-cache behavior across backends.

## Tests

```bash
python3 tests/selftest.py
```

See [tests/README.md](tests/README.md). Releases are cut by merging to `main`; the workflow bumps the version, updates
the changelog and publishes to PyPI.

## License

MIT
