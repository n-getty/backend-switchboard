# backend-switchboard

One local endpoint that lets Claude Code switch between several model backends with `/model`, in a single session:

| `/model` value | Goes to |
|---|---|
| `claude-sonnet-5-5` (no tag) | Anthropic, using your own Claude Code login. Requests are forwarded untouched. |
| `claude-sonnet-5-5@argo` | [Argo](https://github.com/n-getty/argo-shim) through a running `argo-shim` |
| `claude-opus-5-5@google@mag` | A LiteLLM-style gateway with Bearer auth (AMSC "MAG"); the tag `@mag` is stripped and `@google` is MAG's own suffix |
| `beam@alcf`, `muse@alcf` | ALCF inference endpoints (OpenAI chat), translated to the Anthropic API by [llm-rosetta](https://github.com/Oaklight/llm-rosetta) |

Claude Code only accepts one `ANTHROPIC_BASE_URL` per session, so the switch has to happen behind that URL.
backend-switchboard is that URL. It reads the model name, picks a backend, swaps in that backend's credentials, and
streams the response back. Standard library only (Python 3.8+), except for the optional ALCF translation.

> Not affiliated with Anthropic. The Anthropic route forwards your Claude Code login (OAuth or API key) unchanged and
> never reads, stores or logs it. Check your plan's terms before sending subscription traffic through a local proxy.

```
Claude Code --> 127.0.0.1:8790 (backend-switchboard) --+--> api.anthropic.com        (no tag)
   /model name[@tag]                                   +--> argo-shim                (@argo)
                                                       +--> MAG                      (@mag)
                                                       +--> llm-rosetta --> ALCF     (@alcf)
```

## Install

```bash
pip install backend-switchboard            # Anthropic, @argo, @mag
pip install 'backend-switchboard[alcf]'    # adds llm-rosetta for @alcf (Python 3.10+)
# or without installing:
uvx --with 'llm-rosetta==0.7.3' backend-switchboard run
```

## Use

```bash
backend-switchboard run              # starts the router if needed, then launches claude
backend-switchboard run -c           # anything after "run" is passed to claude
```

Then inside Claude Code: `/model beam@alcf`, `/model claude-sonnet-5-5@argo`, `/model claude-sonnet-5-5` (back to your
subscription). You can switch mid-conversation; the history is carried across backends.

Other commands: `init` (create the key and settings), `ensure` (start the router in the background), `stop`, `serve`
(foreground). State lives in `~/.backend-switchboard/` (override with `BACKEND_SWITCHBOARD_HOME`).

If you launch Claude Code with a separate config directory, e.g. a personal subscription, export `CLAUDE_CONFIG_DIR`
first; `run` leaves it alone.

## Backends

**Anthropic (no tag).** Nothing to configure. Whatever login your Claude Code already has is used.

**Argo (`@argo`).** Run [argo-shim](https://github.com/n-getty/argo-shim) as usual. The router re-reads its token and
local URL from `~/.claude/settings.json` on every request (change the path with `argo_settings`). If Argo answers an
exhausted quota with HTTP 200 and an "ACCESS REVOKED" notice, the router turns that into a 429 so it is not shown as a
model answer.

**MAG (`@mag`).** Export `AMSC_TOKEN` in the environment the router starts from. Set `mag_url` in `config.json` to
point at another Bearer-auth gateway. MAG's own provider suffixes stay in the name: `claude-opus-5-5@google@mag`.

**ALCF (`@alcf`).** Install the `alcf` extra, then log in once in your own terminal:

```bash
uv tool install alcf-ai alcf-tokens
alcf-tokens login inference
```

The router starts and supervises an llm-rosetta gateway, and restarts it whenever your ALCF token rotates (the gateway
only reads the token at startup). Aliases `beam` and `muse` map to `beam-1.0-bf16` and `muse-spark-bf16`; any other
name is passed through. Add or change aliases under `aliases.alcf` in `config.json`.

A bare `name@google` is rejected as ambiguous, because more than one backend uses that suffix. Always add a router tag.

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
| `argo_settings` / `argo_url` | `~/.claude/settings.json` / read from it | where to find argo-shim |
| `alcf_url` | ALCF Minerva `/api/v1` | OpenAI-compatible endpoint behind `@alcf` |
| `rosetta_python` | the current interpreter | Python that has llm-rosetta installed |
| `rosetta_port` | `8768` | local port for the llm-rosetta child |
| `alcf_gateway`, `alcf_gateway_key` | unset | use your own Anthropic-compatible gateway for `@alcf` instead of llm-rosetta |
| `aliases` | `beam`, `muse` | short names per backend |

## Security

- The router listens on `127.0.0.1` only and requires a random key on every request, sent by Claude Code through
  `ANTHROPIC_CUSTOM_HEADERS`. This matters on shared login nodes: without it, any other user on the machine could spend
  your Argo, MAG and ALCF credentials. The key and `settings.json` are created with mode 600.
- Upstream credentials are read per request and are never logged. The log records the model name, backend, status and
  duration only.

## Status and limits

- Tested live: your Anthropic login, ALCF Beam and Muse, and a single conversation switched across backends between turns.
  The Argo and MAG routes are covered by mock-upstream tests only.
- Muse did not manage to call a no-argument tool (`CronList`) even with every tool inlined, while Beam did. This is model
  behavior, not routing.
- When a backend is down or returns a 5xx, Claude Code retries for a few minutes, so a stopped Argo tunnel looks like a
  hang. Check `~/.backend-switchboard/router.log`.
- Each ALCF turn includes one request that the gateway answers with HTTP 405 (a token-count call). Claude Code falls back
  and the turn completes.
- Not covered: the interactive `/model` picker UI (typing the name works), and prompt-cache behavior across backends.

## Tests

```bash
python3 tests/selftest.py
```

See [tests/README.md](tests/README.md). Releases are cut by merging to `main`; the workflow bumps the version, updates
the changelog and publishes to PyPI.

## License

MIT
