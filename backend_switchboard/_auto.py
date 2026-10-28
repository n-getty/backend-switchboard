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

TIERS = ("small", "medium", "large")
AUTO_MODEL = "auto"
RETRYABLE = (429, 500, 502, 503, 504, 529)

# Zero-shot policy fitted in llm-router (hardness = offset + sum(weight * d1 feature)); the cut-points are the ones for
# its "luna -> haiku -> opus" ladder. They come from one 600-prompt benchmark, so treat them as a starting point and
# recalibrate from auto.jsonl. A cut of None means "never stop at this tier".
DEFAULTS = {
    "d1_url": None,            # e.g. http://x4712c2s2b0n0:8781 ; None -> heuristic only
    "d1_timeout": 2.0,
    "tiers": {},               # {"small": ["claude-haiku-5-5@mag", "beam@alcf"], "medium": [...], "large": [...]}
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

MAX_CHARS = 16000
SHORT_TURN = 80


def auto_config(cfg):
    out = dict(DEFAULTS)
    out.update(cfg.get("auto") or {})
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


class Plan:
    def __init__(self, tier, candidates, decision):
        self.tier, self.candidates, self.decision = tier, candidates, decision


class Auto:
    def __init__(self, cfg, home):
        self.c = auto_config(cfg)
        self.log_path = os.path.join(home, "auto.jsonl")
        self.cache, self.lock = {}, threading.Lock()
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

    def _hardness_tier(self, f):
        c = self.c
        h = c["offset"] + sum(w * f.get(k, 0.0) for k, w in c["weights"].items())
        lo, hi = TIERS.index(c["floor"]), TIERS.index(c["ceiling"])
        for tier in TIERS[lo:hi + 1]:
            cut = c["cuts"].get(tier)
            if tier == c["ceiling"] or (cut is not None and h < cut + c["quality_bias"]):
                return tier, {"hardness": round(h, 3)}
        return c["ceiling"], {"hardness": round(h, 3)}

    def decide(self, text):
        key = hashlib.sha1(text.encode("utf8", "replace")).hexdigest()
        now = time.time()
        with self.lock:
            hit = self.cache.get(key)
            if hit and now - hit[0] < self.c["cache_ttl"]:
                return dict(hit[1], cached=True)
        t = time.perf_counter()
        source, tier, detail = "heuristic", None, {}
        if self.c["d1_url"] and self.d1.ok("d1"):
            try:
                tier, detail = self._hardness_tier(self._features(text))
                source = "d1"
                self.d1.success("d1")
            except Exception as e:  # never fail the request because routing failed
                self.d1.fail("d1")
                source = "heuristic (%s)" % type(e).__name__
        if tier is None:
            tier, detail = heuristic_tier(text, self.c["quality_bias"])
        d = {"tier": tier, "source": source, "decide_ms": round((time.perf_counter() - t) * 1000, 1),
             "prompt_sha1": key[:12], "prompt_chars": len(text)}
        d.update(detail)
        with self.lock:
            if len(self.cache) > 512:
                self.cache.clear()
            self.cache[key] = (now, d)
        return d

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

    def plan(self, obj):
        if not any(self.c["tiers"].get(t) for t in TIERS):
            return None
        d = self.decide(routing_text(obj))
        return Plan(d["tier"], self.candidates(d["tier"]), d)

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

    def shadow(self, obj, requested):
        """Log what auto would have picked for a request that named a model. Runs off the request path."""
        def run():
            d = self.decide(routing_text(obj))
            cands = self.candidates(d["tier"])
            self.record(shadow=True, requested=requested, tier=d["tier"], would_use=cands[:1], decision=d)
        threading.Thread(target=run, daemon=True).start()
