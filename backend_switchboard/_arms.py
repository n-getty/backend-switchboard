"""Model registry and chooser for dynamic `auto` routing.

An *arm* is one way to answer a request: a router model name (`beam@alcf`), where its quota comes from, what it costs,
what it can do, and how good it is at each kind of task. For every request the chooser scores each eligible arm

    P(success)  = sigmoid(slope * (skill - need))

then takes every arm whose P(success) is within `margin` of the best one and picks the cheapest of them (ties: higher
P). The remaining arms follow by P(success) as the fallback chain. The order is computed per request instead of walking
a fixed small/medium/large ladder: free ALCF arms win whenever they are close enough to the best, a mid-priced model
wins when ALCF is just out of reach, and the expensive model is only chosen when nothing cheaper is near its level.

  skill  : the arm's ability on this request's task mix, minus `unc_weight * unc` (be pessimistic about weak evidence)
  need   : how demanding the request is (0..1), from d1's hardness score
  cost   : prompt size * price_in + expected output * price_out, scaled by the pool's scarcity

Skills are PRIORS read off published numbers (below), not measurements. Different vendors report different benchmark
versions (e.g. Terminal-Bench 2.1 on the Beam and Muse cards, 4.0 on Anthropic's), so cross-family gaps carry a large
`unc`. Replace them with outcome telemetry (auto.jsonl) as it accumulates, or override any field under `auto.arms`.
"""
import math

TASKS = ("terminal", "code", "tools", "science", "knowledge", "chat")

# terminal : shell / long-horizon agentic coding      code     : repo-level edits, general programming
# tools    : MCP / computer-use orchestration         science  : hard reasoning, math, graduate science
# knowledge: factual recall without tools             chat     : conversation, writing, summarising
ARMS = {
    # --- ALCF Minerva: free ------------------------------------------------------------------------------------------
    # Beam 1.0 501B-A23B, BF16 card (reflection/Beam-1.0-501B-A23B): Terminal-Bench 2.1 80.1, SWE-bench Verified 81.6,
    # SWE-Bench Pro 66.1, DeepSWE 43.3, MCP-Atlas 80.8, GPQA-D 90.6, AIME 2026 97.7, HLE 36.0 / 51.1 with tools,
    # SciCode 49.2, AA-Omniscience 10.8, AA-LCR 78.2. Text only. ~345 tok/s on Artificial Analysis.
    "beam": {"model": "beam@alcf", "pool": "alcf", "price": (0.0, 0.0), "vision": False, "ctx": 1000000,
             "skills": (0.66, 0.68, 0.74, 0.76, 0.50, 0.72), "unc": 0.12},
    # Muse Spark Open 816B-A42B, xhigh (meta-models/Muse-Spark-Open-816B-A42B): Terminal-Bench 2.1 76.4, MCP-Atlas 89.2,
    # OSWorld Verified 80.4, Toolathlon 72.8, DeepSWE 60, SWE-Atlas QnA 58, SciCode 58.5, GPQA-D 89.0, HLE 61.0 with
    # tools, AA-Omniscience 25. Image/video/audio input. ~153 tok/s. Whether ALCF's endpoint accepts images is unverified.
    "muse": {"model": "muse@alcf", "pool": "alcf", "price": (0.0, 0.0), "vision": True, "ctx": 1000000,
             "skills": (0.62, 0.72, 0.84, 0.80, 0.70, 0.78), "unc": 0.12},
    # --- Anthropic via Argo (monthly quota; list prices $/M in, out) --------------------------------------------------
    # Anthropic's launch tables: Terminal-Bench 4.0 Haiku 39.2 / Sonnet 70.6 / Opus 66.4; HLE with tools 57.4 / 64.5 /
    # 67.7; OSWorld 2.1 72.4 / 80.1-83.9 / 81.8; FrontierCode 1.1 46.4 / 52.1 / 54.4. Sonnet is the better terminal
    # agent; Opus leads on FrontierCode and HLE.
    "haiku": {"model": "claude-haiku-5-5@argo", "pool": "argo", "price": (0.10, 0.50), "vision": True, "ctx": 1000000,
              "skills": (0.55, 0.75, 0.80, 0.78, 0.65, 0.88), "unc": 0.05},
    "sonnet": {"model": "claude-sonnet-5-5@argo", "pool": "argo", "price": (2.0, 10.0), "vision": True, "ctx": 1000000,
               "skills": (0.90, 0.85, 0.90, 0.88, 0.80, 0.93), "unc": 0.05},
    "opus": {"model": "claude-opus-5-5@argo", "pool": "argo", "price": (4.0, 20.0), "vision": True, "ctx": 1000000,
             "skills": (0.85, 0.90, 0.90, 0.92, 0.85, 0.95), "unc": 0.05},
}

# Multiplier on list price per pool: how scarce that quota is. 0 = free.
# Argo is not charged per token, but its monthly cap is finite (it has been exhausted before), so a token there is worth
# more than its list price; ALCF is free and is spent first.
POOL_SCALE = {"alcf": 0.0, "argo": 3.0, "asksage": 1.0, "mag": 1.0, "anthropic": 1.0}

# Which task a d1 `domain` answer means, as a mix.
DOMAIN_TASKS = {
    "code": {"code": 0.6, "terminal": 0.4},
    "math": {"science": 1.0},
    "science": {"science": 0.7, "knowledge": 0.3},
    "humanities": {"knowledge": 0.7, "chat": 0.3},
    "writing": {"chat": 1.0},
    "chat": {"chat": 1.0},
}
SHELL_TOOLS = ("Bash", "Edit", "Write", "MultiEdit", "NotebookEdit")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def registry(cfg):
    """ARMS merged with config: `auto.arms` overrides fields of a known arm or adds a new one; setting an arm to null
    removes it; `auto.enabled_arms` (a list of names) restricts the set."""
    arms = {k: dict(v) for k, v in ARMS.items()}
    for name, over in (cfg.get("arms") or {}).items():
        if over is None:
            arms.pop(name, None)
        else:
            arms.setdefault(name, {"vision": False, "ctx": 200000, "unc": 0.2, "pool": "argo",
                                   "skills": (0.5,) * len(TASKS)}).update(over)
    if cfg.get("enabled_arms") is not None:
        arms = {k: v for k, v in arms.items() if k in cfg["enabled_arms"]}
    return arms


def task_mix(obj, f):
    """Weights over TASKS: d1's domain answer, plus what the request structure says (tools present -> tool use;
    shell/edit tools present -> terminal work)."""
    dom = {k[7:]: v for k, v in f.items() if k.startswith("domain=")}
    mix = dict.fromkeys(TASKS, 0.0)
    for d, p in (dom or {"chat": 1.0}).items():
        for t, w in DOMAIN_TASKS.get(d, {"chat": 1.0}).items():
            mix[t] += p * w
    tools = obj.get("tools") or []
    names = {t.get("name") for t in tools if isinstance(t, dict)}
    if tools:
        mix["tools"] += 0.3
    if names & set(SHELL_TOOLS):
        mix["terminal"] += 0.3
    total = sum(mix.values()) or 1.0
    return {t: w / total for t, w in mix.items()}


def need(hardness):
    """How demanding the request is, 0..1, from d1's hardness score (-2.3 trivial ... +3.4 very hard)."""
    return 0.10 + 0.80 * sigmoid(1.2 * (hardness + 0.5))


def has_image(obj):
    for m in obj.get("messages", []):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "image" for b in c):
            return True
    return False


def rank(cfg, obj, f, hardness, nbytes, available):
    """Score every eligible arm. Returns a list of dicts, best first. `available` is the set of router backend tags."""
    arms, mix, nd = registry(cfg), task_mix(obj, f), need(hardness)
    tokens_in = nbytes / 3.5
    scale = dict(POOL_SCALE, **(cfg.get("pool_scale") or {}))
    img = has_image(obj)
    out = []
    for name, a in arms.items():
        tag = a["model"].rpartition("@")[2]
        if tag not in available or (img and not a.get("vision")) or tokens_in > 0.9 * a.get("ctx", 200000):
            continue
        skill = sum(mix[t] * s for t, s in zip(TASKS, a["skills"])) - cfg["unc_weight"] * a.get("unc", 0.2)
        p = sigmoid(cfg["slope"] * (skill - nd))
        pin, pout = a["price"]
        k = scale.get(a.get("pool"), 1.0)
        cost = k * (tokens_in * pin + cfg["est_output_tokens"] * pout) / 1e6
        out.append({"arm": name, "model": a["model"], "p": round(p, 3), "cost": round(cost, 4)})
    if not out:
        return out
    floor = max(r["p"] for r in out) - cfg["margin"]
    good = sorted((r for r in out if r["p"] >= floor), key=lambda r: (r["cost"], -r["p"]))
    rest = sorted((r for r in out if r["p"] < floor), key=lambda r: -r["p"])
    return good + rest
