# backend-switchboard

One local endpoint that lets Claude Code switch between several model backends with `/model`, in a single session:

| `/model` value | Goes to |
|---|---|
| `claude-sonnet-5-5` (no tag) | Anthropic, using your own Claude Code login. Requests are forwarded untouched. |
| `claude-sonnet-5-5@argo` | [Argo](https://github.com/n-getty/argo-shim) through a running `argo-shim` |
| `claude-opus-5-5@google@mag` | A LiteLLM-style gateway with Bearer auth (AMSC "MAG"); the tag `@mag` is stripped and `@google` is MAG's own suffix |
| `beam@alcf`, `muse@alcf` | ALCF inference endpoints, which serve the Anthropic Messages API natively |
| `google-claude-opus-5@asksage` | AskSage models, translated in-process to AskSage's `/query` API |

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

**ALCF (`@alcf`).** Install the ALCF token tools and log in once, in your own terminal:

```bash
uv tool install alcf-ai alcf-tokens
alcf-tokens login inference
```

The router runs `alcf-tokens get-token inference` (cached for two minutes), or uses `ALCF_INFERENCE_TOKEN` if you set it.
ALCF's Minerva endpoint accepts the Anthropic Messages API directly with a Bearer token, so requests are forwarded as they
are, including streaming, tool calls and the models' thinking blocks. Aliases `beam` and `muse` map to `beam-1.0-bf16`
and `muse-spark-bf16`; any other name is passed through. Add or change aliases under `aliases.alcf` in `config.json`.

**AskSage (`@asksage`).** Setup:

1. Get an AskSage API key from your AskSage account settings (for ANL: https://api.asksage.anl.gov) and export it in the
   shell that starts the router, e.g. in `~/.bashrc`: `export ASKSAGE_API_KEY=...`. The router reads it per request and
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

A bare `name@google` is rejected as ambiguous, because more than one backend uses that suffix. Always add a router tag.

## Auto routing (`/model auto`)

`/model auto` sends each request to a cheap or a strong backend depending on how hard the user's latest request is.
A small decision model, [LiquidAI d1-3B](https://huggingface.co/LiquidAI/d1-3B), scores the prompt in one forward pass
(no generated tokens); the router turns the score into a tier and takes the first healthy backend configured for it.
Tiers hold ordinary model names, so every tag, alias and credential rule above applies unchanged.

```json
{
  "auto": {
    "d1_url": "http://x4712c2s2b0n0:8781",
    "tiers": {
      "small":  ["claude-haiku-5-5@google@mag", "beam@alcf"],
      "medium": ["claude-sonnet-5-5@argo"],
      "large":  ["claude-opus-5-5@argo", "claude-opus-5-5@google@mag"]
    }
  }
}
```

- **Dynamic mode (no `tiers`).** Instead of a fixed ladder, the router keeps a registry of *arms* (`_arms.py`): Beam and
  Muse on ALCF (free), and Haiku, Sonnet and Opus 5.5 on Argo. Each arm has a price, a quota pool, capabilities (images,
  context size) and a prior skill per task type (terminal work, code, tool use, science, knowledge, chat), read off the
  models' published benchmarks. For each request the router estimates the task mix (d1's domain answer plus the request's
  tools) and how demanding it is (d1's hardness), computes each arm's chance of success, and picks the cheapest arm within
  `margin` (0.08) of the best. The rest, ordered by chance of success, are the fallback chain. Free ALCF arms therefore take
  whatever they are close enough on; Sonnet takes what is just out of their reach; Opus is chosen only when nothing
  cheaper is near its level. The skills are priors, not measurements: vendors report different benchmark versions, so
  cross-family gaps are uncertain (`unc`, subtracted as a penalty). Override or add arms under `auto.arms`, restrict
  them with `auto.enabled_arms`, and scale a pool's scarcity with `auto.pool_scale` (Argo defaults to 3x list price
  because its monthly cap is finite). `decide "prompt" --tools Bash Edit` prints the ranking.
- **Verify and escalate (`auto.verify`, on by default).** Every `auto` answer is held back, checked, and only then sent
  to Claude Code (so `auto` answers arrive whole rather than streaming; explicit model names still stream). Checks that
  always escalate: an empty answer, a call to a tool the request never offered, tool arguments that are not an object
  or miss a required key, and the same tool call repeated a third time (a loop). Checks that escalate once: the model
  says it cannot do the task, or hedges repeatedly. If a stronger arm *also* says it cannot, the router stops and
  delivers that answer: two independent models agreeing is evidence the limit is real, not the model's. The next arm is
  the cheapest untried one with a higher chance of success than the one that just failed (`max_escalations`, default
  2), and after an escalation the stronger arm is kept for the rest of that user turn (`pin_ttl`), so an agent loop does
  not re-try the weak arm on every step. Errors (429/5xx/unreachable) move down the fallback chain instead. The checker
  fails open: if it throws, the answer is delivered. An optional d1 judge (`auto.judge.enabled`, needs the d1
  service's `/judge`) adds a P(correct) / P(refuses) check; it is off until its accuracy has been measured.
- **Synthesis (`auto.synthesis.enabled`, off by default).** On a fresh user turn, when the planner's first choice is one
  of `synthesis.arms` (default Muse and Beam, both free), both are asked in parallel. If either draft makes a tool call
  nothing is merged (a tool call has side effects and cannot be averaged): the better-placed arm's call is used. If both
  are prose and agree (word overlap above `agree_above`), one is delivered. If they differ, `merger` writes the final
  answer from both drafts (`arbiter`, e.g. a paid arm, takes over when they differ a lot). An unusable draft falls back
  to the other. Cost: twice the load on those arms and a longer wait (two reasoning-heavy drafts, then the merge; a
  disagreement took 44 s in one live run), so use it where quality matters more than latency.
- **What d1 reads.** The latest typed user turn, not tool results or Claude Code's injected reminders. A very short
  turn ("yes", "continue") is judged together with the previous one. The decision is cached per prompt, so the many
  requests of one agent loop cost one d1 call.
- **d1 service.** `POST {d1_url}/features` with `{"text": "..."}` returns `{"features": {"expert": ..., "difficulty": ...}}`.
  [`llm_router/d1_service.py`](https://github.com/n-getty/llm-router) is a ready implementation (GPU node, FastAPI).
  If the service is unset, down or slower than `d1_timeout` (2 s), a keyword/length heuristic decides instead; routing
  never fails a request. After three d1 failures it is skipped for 30 s.
- **Fallback.** If the chosen backend answers 429/5xx or cannot be reached before sending anything, the router tries the
  next candidate: the rest of the tier, then higher tiers, then lower ones. A backend that fails 3 times in a row is
  tried last for 60 s. A failure in the middle of a stream cannot be retried.
- **Policy.** `hardness = offset + Σ weight × feature`; a request stays in the first tier whose `cut` it is below
  (`null` = never stop there). `floor`, `ceiling` and `quality_bias` (>0 favours cheaper tiers) adjust it. The defaults
  come from one 600-prompt benchmark and do not transfer between datasets, so calibrate from your own traffic.
- **Telemetry.** `~/.backend-switchboard/auto.jsonl` has one line per decision: tier, hardness, source, candidates tried,
  backend, status, latency. It holds a hash of the prompt, never the text. With `"shadow": true`, requests that name a
  model explicitly are still scored and logged (`shadow: true`, `would_use`) without changing where they go, so you can
  measure the router before trusting it.
- **Try it without sending anything:** `backend-switchboard decide "prove the lemma"` prints the tier and candidate chain.

Not done yet: quota-aware routing (needs per-backend usage accounting). Muse's default `xhigh` reasoning can use up a small
`max_tokens` before it writes any text (the checker sees that as an empty answer); Claude Code's normal budgets are large
enough, but short test requests may need more.

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
| `asksage_url` / `asksage_cert` | `https://api.asksage.anl.gov/server` / none | AskSage server and optional extra CA bundle |
| `alcf_url` | ALCF Minerva `/api` | endpoint behind `@alcf` (Anthropic Messages API) |
| `aliases` | `beam`, `muse` | short names per backend |
| `auto` | unset | enables `/model auto`; see [Auto routing](#auto-routing-model-auto) |

Set `BACKEND_SWITCHBOARD_DEBUG=1` to print tracebacks for router errors.

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
