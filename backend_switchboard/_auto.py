"""Difficulty-based routing for the `auto` model name.

`/model auto` asks a small decision model (LiquidAI d1-3B, served separately; see README) how hard the user's latest
request is, maps that to a tier (small / medium / large), and picks the first healthy backend configured for the tier.
Backends are the router's normal `name[@tag]` model names, so every credential and alias rule in `resolve()` applies.

  decision  : d1 service features (zero-shot hardness score) -> tier. If d1 is down or slow, a keyword/length
              heuristic decides instead; routing never fails a request.
  fallback  : if the chosen backend errors before it has sent a byte (429/5xx/connect), the next candidate is tried:
              the rest of the tier, then higher tiers, then lower ones. A backend that keeps failing is skipped for
              a cooldown (circuit breaker).
  telemetry : one JSON line per decision in $BACKEND_SWITCHBOARD_HOME/auto.jsonl. Prompt text is never written,
              only a hash, so the log can be shared for calibration.

Standard library only.
"""
import hashlib
import http.client
import json
import os
import re
import threading
import time
from urllib.parse import urlsplit

from . import _arms

TIERS = ("small", "medium", "large")
AUTO_MODEL = "auto"
RETRYABLE = (429, 500, 502, 503, 504, 529)

# Zero-shot policy fitted in llm-router (hardness = offset + sum(weight * d1 feature)); the cut-points are the ones for
# its "luna -> haiku -> opus" ladder. They come from one 600-prompt benchmark, so treat them as a starting point and
# recalibrate from auto.jsonl. A cut of None means "never stop at this tier".
DEFAULTS = {
    "d1_url": None,            # e.g. http://x4712c2s2b0n0:8781 ; None -> heuristic only
    "d1_timeout": 2.0,
    "tiers": {},               # fixed ladder {"small": [...], "medium": [...], "large": [...]}; if empty, arms are used
    "arms": {},                # override/extend the built-in model registry (see _arms.py)
    "enabled_arms": None,      # restrict the registry to these arm names
    "margin": 0.08,            # accept the cheapest arm whose P(success) is within this of the best: higher = cheaper
    "slope": 8.0,              # how sharply P(success) falls as a request outgrows an arm's skill
    "unc_weight": 1.0,         # how much to distrust weakly-evidenced skills
    "est_output_tokens": 1500,
    "pool_scale": {},          # per-pool scarcity multiplier on list price, e.g. {"argo": 2.0}
    "verify": True,            # buffer each auto answer, check it, and escalate a weak one to a stronger arm
    "max_escalations": 2,
    "pin_ttl": 600,            # after an escalation, keep the stronger arm for the rest of that user turn (seconds)
    "judge": {"enabled": False, "min_correct": 0.5, "max_refuses": 0.5},   # d1 /judge as an extra verifier
    # Ask two arms in parallel and merge their answers. Off by default: it doubles load on the arms involved.
    "synthesis": {"enabled": False, "arms": ["muse", "beam"], "merger": "muse", "min_need": 0.35,
                  "agree_above": 0.6, "arbiter": None},
    "floor": "small",
    "ceiling": "large",
    "weights": {"expert": 3.752659678844416, "difficulty": 2.2941577300354083},
    "offset": -4.2172713794406835,
    "cuts": {"small": -0.6328411692004681, "medium": -0.3002796273648851},
    "quality_bias": 0.0,       # >0 sends more traffic to cheaper tiers, <0 escalates more
    "shadow": False,           # also log what auto WOULD pick for requests that name a model explicitly
    "cache_ttl": 600,          # one d1 call per user turn: tool-loop requests repeat the same prompt
    "breaker_failures": 3,
    "breaker_cooldown": 60,
}

NESTED = ("judge", "synthesis")
MAX_CHARS = 16000
SHORT_TURN = 80


def auto_config(cfg):
    """DEFAULTS overlaid with config["auto"]; the nested option dicts merge key by key, so a partial
    "synthesis": {"enabled": true} keeps the other defaults."""
    out = dict(DEFAULTS)
    for k, v in (cfg.get("auto") or {}).items():
        out[k] = dict(out[k], **v) if isinstance(out.get(k), dict) and isinstance(v, dict) and k in NESTED else v
    return out


def wants(model):
    return model == AUTO_MODEL


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"
                         and not b.get("text", "").lstrip().startswith("<system-reminder>"))
    return ""


def routing_text(obj):
    """The user's latest typed request (tool results and Claude Code's injected reminders excluded). A very short turn
    ("yes", "continue") carries no difficulty signal, so the previous typed turn is prepended."""
    turns = [t for t in (_text_of(m.get("content")).strip() for m in obj.get("messages", [])
                         if isinstance(m, dict) and m.get("role") == "user") if t]
    if not turns:
        return ""
    text = turns[-1]
    if len(text) < SHORT_TURN and len(turns) > 1:
        text = turns[-2] + "\n\n" + text
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS // 2] + "\n...\n" + text[-MAX_CHARS // 2:]
    return text


_HARD = re.compile(r"\b(prove|proof|derive|theorem|integral|optimi[sz]e|complexity|concurren|race condition|"
                   r"distributed|refactor|architecture|lemma|asymptotic|eigen|differential|quantum)\b", re.I)
_CODE = re.compile(r"```|\bdef |\bclass |#include|\bfunction\b|\bimport |Traceback|stack trace", re.I)


def heuristic_tier(text, bias=0.0):
    """Fallback when d1 is unavailable: length + keyword cues."""
    n, hard = len(text), len(_HARD.findall(text))
    code = 1.0 if _CODE.search(text) else 0.0
    large = min(1.0, 0.15 + 0.2 * hard + (0.2 if n > 4000 else 0))
    small = max(0.0, 0.7 - 0.25 * hard - 0.3 * code - (0.3 if n > 2000 else 0))
    if small > 0.5 - bias:
        return "small", {"small": round(small, 3), "large": round(large, 3)}
    if large < 0.5 + bias:
        return "medium", {"small": round(small, 3), "large": round(large, 3)}
    return "large", {"small": round(small, 3), "large": round(large, 3)}


class Breaker:
    """Skip a failing target for `cooldown` seconds after `failures` consecutive failures."""

    def __init__(self, failures, cooldown):
        self.failures, self.cooldown = failures, cooldown
        self.n, self.until, self.lock = {}, {}, threading.Lock()

    def ok(self, key):
        with self.lock:
            return time.time() >= self.until.get(key, 0.0)

    def success(self, key):
        with self.lock:
            self.n[key] = 0

    def fail(self, key):
        with self.lock:
            self.n[key] = self.n.get(key, 0) + 1
            if self.n[key] >= self.failures:
                self.n[key], self.until[key] = 0, time.time() + self.cooldown


def heuristic_features(text, tier):
    """d1-shaped features for the arms chooser when d1 is unavailable."""
    code = 1.0 if _CODE.search(text) else 0.0
    return {"_hardness": {"small": -1.5, "medium": -0.45, "large": 2.0}[tier],
            "domain=code": code, "domain=chat": 1.0 - code}


class Plan:
    def __init__(self, tier, candidates, decision, ranked=None):
        self.tier, self.candidates, self.decision = tier, candidates, decision
        self.ranked = ranked or []       # arms mode: every eligible arm with its P(success) and cost
        self.need = decision.get("need", 0.0)


class Auto:
    def __init__(self, cfg, home, backends=()):
        self.backends_available = set(backends)
        self.c = auto_config(cfg)
        self.log_path = os.path.join(home, "auto.jsonl")
        self.cache, self.pins, self.lock = {}, {}, threading.Lock()
        self.backends = Breaker(self.c["breaker_failures"], self.c["breaker_cooldown"])
        self.d1 = Breaker(3, 30)  # d1 itself: after 3 failures stop paying its timeout on every request

    def warm_up(self):
        """The first d1 forward pass after the service starts takes seconds (kernel compile), longer than d1_timeout,
        so without this the first real request would fall back to the heuristic. Best effort, off the request path."""
        if not self.c["d1_url"]:
            return

        def run():
            try:
                self._features("warm up")
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    # --- decision -------------------------------------------------------------------------------------------------
    def _features(self, text):
        u = urlsplit(self.c["d1_url"])
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.c["d1_timeout"])
        try:
            conn.request("POST", "/features", json.dumps({"text": text}), {"content-type": "application/json"})
            r = conn.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError("d1 service HTTP %d" % r.status)
            return json.loads(body)["features"]
        finally:
            conn.close()

    def hardness(self, f):
        if "_hardness" in f:
            return f["_hardness"]
        return self.c["offset"] + sum(w * f.get(k, 0.0) for k, w in self.c["weights"].items())

    def _hardness_tier(self, f):
        c = self.c
        h = self.hardness(f)
        lo, hi = TIERS.index(c["floor"]), TIERS.index(c["ceiling"])
        for tier in TIERS[lo:hi + 1]:
            cut = c["cuts"].get(tier)
            if tier == c["ceiling"] or (cut is not None and h < cut + c["quality_bias"]):
                return tier, {"hardness": round(h, 3)}
        return c["ceiling"], {"hardness": round(h, 3)}

    def decide(self, text):
        return self.assess(text)[0]

    def assess(self, text):
        """-> (decision, d1 features). One d1 call per distinct prompt (cached)."""
        key = hashlib.sha1(text.encode("utf8", "replace")).hexdigest()
        now = time.time()
        with self.lock:
            hit = self.cache.get(key)
            if hit and now - hit[0] < self.c["cache_ttl"]:
                return dict(hit[1], cached=True), hit[2]
        t = time.perf_counter()
        source, tier, detail, f = "heuristic", None, {}, None
        if self.c["d1_url"] and self.d1.ok("d1"):
            try:
                f = self._features(text)
                tier, detail = self._hardness_tier(f)
                source = "d1"
                self.d1.success("d1")
            except Exception as e:  # never fail the request because routing failed
                self.d1.fail("d1")
                source = "heuristic (%s)" % type(e).__name__
        if tier is None:
            tier, detail = heuristic_tier(text, self.c["quality_bias"])
            f = heuristic_features(text, tier)
        d = {"tier": tier, "source": source, "decide_ms": round((time.perf_counter() - t) * 1000, 1),
             "prompt_sha1": key[:12], "prompt_chars": len(text)}
        d.update(detail)
        with self.lock:
            if len(self.cache) > 512:
                self.cache.clear()
            self.cache[key] = (now, d, f)
        return d, f

    # --- candidates -----------------------------------------------------------------------------------------------
    def candidates(self, tier):
        """Chosen tier first, then higher tiers, then lower ones; backends with an open breaker go last."""
        tiers = self.c["tiers"]
        i = TIERS.index(tier)
        order = [tier] + list(TIERS[i + 1:]) + list(reversed(TIERS[:i]))
        seen, out = set(), []
        for t in order:
            for m in tiers.get(t) or []:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
        healthy = [m for m in out if self.backends.ok(m)]
        return healthy + [m for m in out if m not in healthy]

    def arms(self, obj, text, nbytes=0):
        """Rank the registry for this request. Returns (decision, ranked arms) or (decision, [])."""
        d, f = self.assess(text)
        ranked = _arms.rank(self.c, obj, f, self.hardness(f), nbytes, self.backends_available)
        d = dict(d, mix={t: round(w, 2) for t, w in _arms.task_mix(obj, f).items() if w >= 0.05},
                 need=round(_arms.need(self.hardness(f)), 2), ranked=ranked[:5])
        return d, ranked

    def plan(self, obj, nbytes=0):
        text = routing_text(obj)
        if any(self.c["tiers"].get(t) for t in TIERS):
            d = self.decide(text)
            return Plan(d["tier"], self._pinned_first(d["prompt_sha1"], self.candidates(d["tier"])), d)
        d, ranked = self.arms(obj, text, nbytes)
        if not ranked:
            return None
        models = [r["model"] for r in ranked]
        healthy = [m for m in models if self.backends.ok(m)]
        order = self._pinned_first(d["prompt_sha1"], healthy + [m for m in models if m not in healthy])
        return Plan(ranked[0]["arm"], order, d, ranked)

    # --- escalation -----------------------------------------------------------------------------------------------
    def _pinned_first(self, sha, order):
        with self.lock:
            hit = self.pins.get(sha)
        if hit and time.time() < hit[1] and hit[0] in order:
            return [hit[0]] + [m for m in order if m != hit[0]]
        return order

    def pin(self, plan, model):
        """Keep `model` for the rest of this user turn: the next requests of the same agent loop skip the weak arms."""
        with self.lock:
            if len(self.pins) > 512:
                self.pins.clear()
            self.pins[plan.decision["prompt_sha1"]] = (model, time.time() + self.c["pin_ttl"])

    def next_candidate(self, plan, tried, failed, hard_failure):
        """Pick the arm to try after `failed` gave a weak answer (or an error: hard_failure=False).

        An error moves down the fallback chain. A weak answer must go to something stronger: the cheapest untried arm
        with a higher chance of success than the one that just failed (arms mode), or the next candidate up the ladder
        (tiers mode). Returns a router model name, or None when nothing stronger is left."""
        untried = [m for m in plan.candidates if m not in tried]
        if not untried:
            return None
        if not hard_failure or not plan.ranked:
            return untried[0]
        p_of = {r["model"]: r["p"] for r in plan.ranked}
        base = p_of.get(failed, 0.0)
        stronger = [r for r in plan.ranked if r["model"] in untried and r["p"] > base + 0.02]
        if not stronger:
            return None
        return min(stronger, key=lambda r: (r["cost"], -r["p"]))["model"]

    # --- optional d1 judge ----------------------------------------------------------------------------------------
    def judge(self, request_text, response_text):
        """P(correct)/P(refuses) for an answer from the d1 service's /judge endpoint, or {} if unavailable."""
        j = self.c["judge"]
        if not (j.get("enabled") and self.c["d1_url"]) or not self.d1.ok("judge"):
            return {}
        u = urlsplit(self.c["d1_url"])
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.c["d1_timeout"] * 2)
        try:
            conn.request("POST", "/judge", json.dumps({"request": request_text, "response": response_text}),
                         {"content-type": "application/json"})
            r = conn.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError("judge HTTP %d" % r.status)
            self.d1.success("judge")
            return json.loads(body)["features"]
        except Exception:
            self.d1.fail("judge")
            return {}
        finally:
            conn.close()

    # --- telemetry ------------------------------------------------------------------------------------------------
    def record(self, **fields):
        fields["ts"] = round(time.time(), 3)
        try:
            with self.lock:
                fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as f:
                    f.write(json.dumps(fields) + "\n")
        except OSError:
            pass

    def shadow(self, obj, requested, nbytes=0):
        """Log what auto would have picked for a request that named a model. Runs off the request path."""
        def run():
            p = self.plan(obj, nbytes)
            if p:
                self.record(shadow=True, requested=requested, tier=p.tier, would_use=p.candidates[:1],
                            decision=p.decision)
        threading.Thread(target=run, daemon=True).start()
