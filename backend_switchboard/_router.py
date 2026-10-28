#!/usr/bin/env python3
"""backend-switchboard: one local Anthropic-API endpoint for Claude Code, routed by /model suffix.

  claude-opus-5-5              -> Anthropic (your subscription; client headers pass through untouched)
  claude-opus-5-5@google@mag   -> AMSC MAG   (router tag @mag is stripped; "@google" is MAG's own suffix)
  claude-opus-5-5@argo         -> Argo via argo-shim
  beam@alcf / muse@alcf        -> ALCF Minerva (serves the Anthropic Messages API natively; Bearer token)
  gpt-5.4@asksage              -> AskSage /query (translated in-process; needs ASKSAGE_API_KEY)
  auto                         -> whichever of the above config.json "auto" (arms or tiers) picks for the request

Standard library only (3.8+). Upstream credentials are read per request and never logged.
"""
import argparse
import hmac
import http.client
import json
import os
import secrets
import shlex
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import _asksage as asksage
from . import _auto as auto_mod
from . import _verify as verify

HOME = os.path.expanduser("~")
DIR = os.environ.get("BACKEND_SWITCHBOARD_HOME", os.path.join(HOME, ".backend-switchboard"))
BACKENDS = ("anthropic", "argo", "mag", "alcf", "asksage")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
SOFT_ERRORS = (b"ACCESS REVOKED", b"IMPORTANT USAGE NOTICE")

DEFAULTS = {
    "port": 8790,
    "tool_search": "true",
    "anthropic_url": "https://api.anthropic.com",
    "mag_url": "https://mag.genesis.americansciencecloud.org",
    "argo_settings": os.path.join(HOME, ".claude", "settings.json"),
    "argo_url": None,
    "alcf_url": "https://inference-api.alcf.anl.gov/resource_server/minerva/api",
    "asksage_url": "https://api.asksage.anl.gov/server",
    "asksage_cert": None,  # extra CA bundle (ANL's AskSage cert), e.g. ~/.config/asksage_proxy/asksage_anl_gov.pem
    "aliases": {"alcf": {"beam": "beam-1.0-bf16", "muse": "muse-spark-bf16"}},
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(os.path.join(DIR, "config.json")) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    return cfg


def read_key():
    with open(os.path.join(DIR, "key")) as f:
        return f.read().strip()


def split_model(model):
    """'name@mag' -> ('name', 'mag'); only the LAST @tag, and only if it is a router backend."""
    base, sep, tag = model.rpartition("@")
    if sep and tag in BACKENDS:
        return base, tag
    return model, None


class RouteError(Exception):
    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status, self.message = status, message


_cache = {}


def cached_file(path, parse):
    """Re-parse a file only when its mtime changes (tokens rotate; settings files get rewritten)."""
    try:
        m = os.stat(path).st_mtime
    except OSError:
        return None
    key = (path, parse.__name__)
    hit = _cache.get(key)
    if hit and hit[0] == m:
        return hit[1]
    try:
        val = parse(path)
    except Exception:
        val = None
    _cache[key] = (m, val)
    return val


def _argo_token(path):
    helper = json.load(open(path)).get("apiKeyHelper", "")
    words = shlex.split(helper)
    return words[-1] if words else None


def _argo_base(path):
    url = json.load(open(path)).get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return url if urlsplit(url).hostname in ("127.0.0.1", "localhost") else None


def mag_token():
    return os.environ.get("AMSC_TOKEN")


_alcf = {"lock": threading.Lock(), "token": None, "at": 0.0}


def alcf_token():
    """ALCF inference token: ALCF_INFERENCE_TOKEN if set, else `alcf-tokens get-token inference` (cached 2 min)."""
    tok = os.environ.get("ALCF_INFERENCE_TOKEN")
    if tok:
        return tok
    with _alcf["lock"]:
        if _alcf["token"] and time.time() - _alcf["at"] < 120:
            return _alcf["token"]
        exe = shutil.which("alcf-tokens") or os.path.join(HOME, ".local/bin/alcf-tokens")
        try:
            out = subprocess.run([exe, "get-token", "inference"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        _alcf["token"], _alcf["at"] = out.stdout.strip() or None, time.time()
        return _alcf["token"]


def resolve(cfg, model):
    """Return (base_url, extra_headers, drop_client_auth, upstream_model, backend)."""
    base, tag = split_model(model)
    if tag is None:
        if "@" in model:
            raise RouteError(400, "ambiguous model %r: add a router tag, e.g. %s@mag or %s@argo (tags: %s)"
                             % (model, model, model, ", ".join(BACKENDS)))
        return cfg["anthropic_url"], {}, False, model, "anthropic"
    if tag == "anthropic":
        return cfg["anthropic_url"], {}, False, base, tag
    if tag == "mag":
        tok = mag_token()
        if not tok:
            raise RouteError(401, "AMSC_TOKEN is not set in the router's environment (restart it after exporting)")
        return cfg["mag_url"], {"authorization": "Bearer " + tok}, True, base, tag
    if tag == "argo":
        tok = cached_file(cfg["argo_settings"], _argo_token)
        url = cfg["argo_url"] or cached_file(cfg["argo_settings"], _argo_base)
        if not tok or not url:
            raise RouteError(502, "cannot read argo-shim token/base URL from %s" % cfg["argo_settings"])
        return url, {"x-api-key": tok}, True, base, tag
    if tag == "asksage":
        return cfg["asksage_url"], {}, True, base, tag
    tok = alcf_token()
    if not tok:
        raise RouteError(401, "no ALCF token; run in your own terminal: alcf-tokens login inference")
    return cfg["alcf_url"], {"authorization": "Bearer " + tok}, True, cfg["aliases"]["alcf"].get(base, base), tag


BLOCK_TYPES = {"text", "image", "document", "tool_use", "tool_result", "thinking", "redacted_thinking",
               "server_tool_use", "web_search_tool_result", "search_result", "container_upload"}


def strip_client_extensions(obj):
    """Drop Claude Code's Anthropic-only extras that Vertex-backed Argo rejects with a 400: message-level keys other
    than role/content (e.g. output_config) and content blocks of unknown type (e.g. tool_addition).
    Returns True if anything changed."""
    changed = False
    for msg in obj.get("messages", []):
        if not isinstance(msg, dict):
            continue
        for k in [k for k in msg if k not in ("role", "content")]:
            del msg[k]
            changed = True
        content = msg.get("content")
        if isinstance(content, list):
            kept = [b for b in content if not isinstance(b, dict) or b.get("type") in BLOCK_TYPES]
            if len(kept) != len(content):
                msg["content"] = kept
                changed = True
    return changed


def flatten_tool_search(obj):
    """Make Claude Code's deferred-tool search safe for backends that lack Anthropic's tool_reference blocks.

    The client already adds a loaded tool's full schema to `tools` on the next request, so it is enough to drop
    the defer_loading placeholder and turn each tool_reference into a text result. Returns True if anything changed.
    """
    changed = False
    tools = obj.get("tools")
    if isinstance(tools, list):
        kept = [t for t in tools if not (isinstance(t, dict) and t.get("defer_loading"))]
        changed = len(kept) != len(tools)
        obj["tools"] = kept
    for msg in obj.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for blk in content:
            inner = blk.get("content") if isinstance(blk, dict) and blk.get("type") == "tool_result" else None
            if not isinstance(inner, list):
                continue
            for i, b in enumerate(inner):
                if isinstance(b, dict) and b.get("type") == "tool_reference":
                    inner[i] = {"type": "text", "text": "Tool loaded: %s. Its schema is now in your tool list; "
                                                        "call it directly." % b.get("tool_name")}
                    changed = True
    return changed


class Result:
    """A buffered upstream answer, held back until `auto` has checked it."""

    def __init__(self, status, headers, body, backend):
        self.status, self.headers, self.body, self.backend = status, headers, body, backend

    @property
    def ctype(self):
        return next((v for k, v in self.headers if k.lower() == "content-type"), "")


class Capture:
    """Stands in for the HTTP handler so a backend that writes its own response (AskSage) can be buffered."""

    def __init__(self, h):
        self.path, self.command, self.headers = h.path, h.command, h.headers
        self.status, self.headers_out, self.close_connection, self.body = 200, [], False, bytearray()
        self.wfile = self

    def send_response(self, status, reason=None):
        self.status = status

    def send_header(self, k, v):
        self.headers_out.append((k, v))

    def end_headers(self):
        pass

    def write(self, b):
        self.body += b

    def flush(self):
        pass


def _error_body(status, message):
    return json.dumps({"type": "error", "error": {"type": "invalid_request_error" if status < 500 else "api_error",
                                                   "message": message}}).encode()


SYNTH_PROMPT = ("Two independent drafts answer the request above. Write the single best final answer to it. Keep what "
                "the drafts agree on; where they disagree, work out which is right; fix any mistakes. Do not mention "
                "the drafts or that you were given them.\n\n<draft_1>\n%s\n</draft_1>\n\n<draft_2>\n%s\n</draft_2>")


def last_turn_is_typed(obj):
    """True if the newest message is something the user typed (not just tool results returning to the model)."""
    msgs = obj.get("messages") or []
    if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") != "user":
        return False
    c = msgs[-1].get("content")
    return isinstance(c, str) or (isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "text" for b in c))


def synth_request(obj, drafts):
    """The original request with no tools and both drafts appended to the user's last message."""
    o = json.loads(json.dumps(obj))
    o.pop("tools", None)
    o.pop("tool_choice", None)
    last = o["messages"][-1]
    blocks = [{"type": "text", "text": last["content"]}] if isinstance(last["content"], str) else last["content"]
    blocks.append({"type": "text", "text": SYNTH_PROMPT % tuple(drafts)})
    last["content"] = blocks
    return o


class Handler(BaseHTTPRequestHandler):
    cfg = key = auto = None
    server_version = "backend-switchboard"

    def log_message(self, *a):
        pass

    def _err(self, status, message):
        body = _error_body(status, message)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            return self._plain(b'{"ok":true}')
        self._proxy()

    def do_POST(self):
        self._proxy()

    def _plain(self, body):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self):
        t0 = time.time()
        if not hmac.compare_digest(self.headers.get("x-router-key", ""), self.key):
            return self._err(401, "missing or wrong x-router-key")
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n) if n else b""
        model, backend, plan, chosen, attempts, info = "-", "?", None, None, [], {}
        try:
            obj = None
            if raw and self.path.split("?")[0].startswith("/v1/messages"):
                try:
                    obj = json.loads(raw)
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and isinstance(obj.get("model"), str):
                    model = obj["model"]
            if self.auto and model != "-":
                if auto_mod.wants(model):
                    plan = self.auto.plan(obj, len(raw))
                    if plan is None:
                        raise RouteError(400, "model 'auto' has no eligible backend (check auto.tiers / auto.arms / "
                                              "auto.enabled_arms in %s/config.json)" % DIR)
                elif self.auto.c["shadow"]:
                    self.auto.shadow(obj, model, len(raw))
            if plan is None:
                chosen = model
                status, backend = self._route_once(raw, obj, model)
            else:
                res, chosen = self._run_auto(raw, obj, plan, attempts, info)
                self._deliver(res)
                status, backend = res.status, res.backend
        except RouteError as e:
            status = e.status
            self._err(e.status, e.message)
        except Exception as e:
            status = 502 if isinstance(e, (OSError, http.client.HTTPException)) else 500
            sys.stderr.write("router error: %r\n" % (e,))
            if os.environ.get("BACKEND_SWITCHBOARD_DEBUG"):
                traceback.print_exc()
            try:
                self._err(status, "router/upstream error: %s" % e)
            except OSError:
                pass
        label = "%s>%s" % (model, chosen) if plan else model
        if plan:
            self.auto.record(requested=model, tier=plan.tier, decision=plan.decision, attempts=attempts,
                             chosen=chosen, backend=backend, status=status, total_ms=round((time.time() - t0) * 1000),
                             **info)
        sys.stderr.write("%s %-40s -> %s %s %.2fs\n" % (time.strftime("%H:%M:%S"), label[:40],
                         backend, status, time.time() - t0))
        sys.stderr.flush()

    # --- one candidate ----------------------------------------------------------------------------------------------
    def _prepare(self, raw, obj, model):
        """Resolve `model` and rewrite the request for it -> (url, extra, drop_auth, body, backend, obj)."""
        body = raw
        if model == "-":
            return self.cfg["anthropic_url"], {}, False, body, "anthropic", obj
        url, extra, drop_auth, up_model, backend = resolve(self.cfg, model)
        dirty = up_model != obj["model"]
        obj["model"] = up_model
        if backend != "anthropic":
            dirty = flatten_tool_search(obj) or dirty
            dirty = strip_client_extensions(obj) or dirty
        if dirty:
            body = json.dumps(obj).encode()
        return url, extra, drop_auth, body, backend, obj

    def _route_once(self, raw, obj, model):
        """Forward to `model`, streaming the answer straight to the client. Returns (status, backend)."""
        url, extra, drop_auth, body, backend, obj = self._prepare(raw, obj, model)
        if backend == "asksage":
            return asksage.handle(self, self.cfg, obj), backend
        return self._forward(url, extra, drop_auth, body, backend), backend

    def _upstream(self, url, extra, drop_auth, backend, strip_encoding=False):
        u = urlsplit(url)
        headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in HOP or lk == "x-router-key":
                continue
            if drop_auth and lk in ("authorization", "x-api-key"):
                continue
            if lk == "accept-encoding" and (backend == "argo" or strip_encoding):
                continue
            headers[lk] = v
        headers.update(extra)
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=900, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
        return conn, u.path.rstrip("/") + self.path, headers

    def _forward(self, url, extra, drop_auth, body, backend):
        conn, path, headers = self._upstream(url, extra, drop_auth, backend)
        conn.request(self.command, path, body or None, headers)
        resp = conn.getresponse()
        first = b""
        if backend == "argo" and resp.status == 200:
            first = resp.read1(8192)
            if any(s in first[:4000] for s in SOFT_ERRORS):
                self._err(429, "Argo returned a usage/quota notice (HTTP 200 soft error): " + first[:300].decode("utf8", "replace"))
                conn.close()
                return 429
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        if first:
            self.wfile.write(first)
            self.wfile.flush()
        while True:
            chunk = resp.read1(8192)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        conn.close()
        return resp.status

    def _fetch(self, raw, obj, model):
        """Like _route_once, but hold the whole answer back and return it as a Result. Raises RouteError."""
        url, extra, drop_auth, body, backend, obj = self._prepare(raw, obj, model)
        if backend == "asksage":
            cap = Capture(self)
            asksage.handle(cap, self.cfg, obj)
            return Result(cap.status, cap.headers_out, bytes(cap.body), backend)
        conn, path, headers = self._upstream(url, extra, drop_auth, backend, strip_encoding=True)
        try:
            conn.request(self.command, path, body or None, headers)
            resp = conn.getresponse()
            data = resp.read()
            res = Result(resp.status, resp.getheaders(), data, backend)
        except (OSError, http.client.HTTPException) as e:
            return Result(502, [("content-type", "application/json")],
                          _error_body(502, "%s unreachable: %s" % (backend, e)), backend)
        finally:
            conn.close()
        if backend == "argo" and res.status == 200 and any(s in data[:4000] for s in SOFT_ERRORS):
            return Result(429, [("content-type", "application/json")],
                          _error_body(429, "Argo returned a usage/quota notice (HTTP 200 soft error): "
                                      + data[:300].decode("utf8", "replace")), backend)
        return res

    def _deliver(self, res):
        self.send_response(res.status)
        for k, v in res.headers:
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("content-length", str(len(res.body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(res.body)
        self.close_connection = True

    # --- auto: fetch, check, escalate -------------------------------------------------------------------------------
    def _issues(self, res, obj, text):
        """What is wrong with an answer, as (code, severity, detail) tuples; [] when it is fine."""
        msg = verify.parse(res.body, res.ctype)
        issues = verify.check(msg, obj)
        lim = self.auto.c["judge"]
        if not issues and msg and msg["text"].strip() and not msg["tool_uses"] and len(msg["text"]) <= lim.get("max_chars", 2000):
            j = self.auto.judge(text, msg["text"])
            if j.get("correct") is not None and j["correct"] < lim["min_correct"]:
                issues.append(("judge_low", "soft", "judge P(correct)=%.2f" % j["correct"]))
            if j.get("refuses") is not None and j["refuses"] > lim["max_refuses"]:
                issues.append(("judge_refuses", "soft", "judge P(refuses)=%.2f" % j["refuses"]))
        return issues

    def _run_auto(self, raw, obj, plan, attempts, info):
        """Try the planned arms in order. Each 200 answer is checked; a weak one escalates to a stronger arm, an error
        moves down the fallback chain. Returns (Result, model that produced it)."""
        a, c = self.auto, self.auto.c
        syn = c["synthesis"]
        if syn["enabled"] and plan.ranked:
            out = self._synthesize(raw, obj, plan, attempts, info)
            if out:
                return out
        text = auto_mod.routing_text(obj)
        tried, escalations, fallback, last_err = [], 0, None, None
        cand = plan.candidates[0]
        while cand is not None:
            tried.append(cand)
            try:
                r = self._fetch(raw, json.loads(raw), cand)
            except RouteError as e:
                attempts.append([cand, e.status])
                last_err, cand = e, a.next_candidate(plan, tried, cand, False)
                continue
            if r.status in auto_mod.RETRYABLE:
                a.backends.fail(cand)
                attempts.append([cand, r.status])
                fallback = fallback or r
                cand = a.next_candidate(plan, tried, cand, False)
                continue
            a.backends.success(cand)
            if r.status != 200 or not c["verify"]:
                return r, cand
            try:
                issues = self._issues(r, obj, text)
            except Exception as e:  # the checker must never break a request: accept what we got
                sys.stderr.write("verify error: %r\n" % (e,))
                issues = []
            if not issues:
                if escalations:
                    a.pin(plan, cand)
                return r, cand
            attempts.append([cand, r.status, [i[0] for i in issues]])
            fallback = r
            if escalations and all(i[1] == "soft" for i in issues):
                # A stronger arm also says it can't / isn't sure: that is evidence the limit is real, not the model's.
                return r, cand
            nxt = a.next_candidate(plan, tried, cand, True) if escalations < c["max_escalations"] else None
            if nxt is None:
                return r, cand
            escalations += 1
            info["escalations"] = escalations
            cand = nxt
        if fallback is not None:
            return fallback, tried[-1]
        raise last_err or RouteError(502, "no backend could answer")

    def _synthesize(self, raw, obj, plan, attempts, info):
        """Ask two arms in parallel and combine. Returns (Result, model) or None to fall back to the normal path.

        Only on a fresh user turn and when the planner's first choice is one of the synthesis arms (the cheap pool is
        judged good enough). If either draft makes a tool call there is nothing to merge (a tool call has side effects
        and cannot be averaged), so one draft is chosen; if both are prose and agree, the better-placed arm's draft is
        used; if they differ, a merger call writes the final answer."""
        syn, a = self.auto.c["synthesis"], self.auto
        arms = auto_mod._arms.registry(a.c)
        models = [arms[n]["model"] for n in syn["arms"] if n in arms and arms[n]["model"] in plan.candidates]
        if (len(models) < 2 or plan.candidates[0] not in models or plan.need < syn["min_need"]
                or not last_turn_is_typed(obj)):
            return None
        results = {}

        def run(m):
            try:
                results[m] = self._fetch(raw, json.loads(raw), m)
            except RouteError as e:
                results[m] = Result(e.status, [], _error_body(e.status, e.message), "?")
        threads = [threading.Thread(target=run, args=(m,)) for m in models[:2]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        p_of = {r["model"]: r["p"] for r in plan.ranked}
        text, good = auto_mod.routing_text(obj), []
        for m in models[:2]:
            r = results[m]
            msg = verify.parse(r.body, r.ctype) if r.status == 200 else None
            hard = [i for i in verify.check(msg, obj) if i[1] == "hard"] if msg else [("unparsed", "hard", "")]
            attempts.append([m, r.status, "draft"] + [i[0] for i in hard])
            (a.backends.success if r.status not in auto_mod.RETRYABLE else a.backends.fail)(m)
            if r.status == 200 and not hard:
                good.append((m, r, msg))
        info["synthesis"] = len(good)
        if len(good) < 2:
            if not good:
                return None
            info["synthesis"] = "single"
            return good[0][1], good[0][0]
        good.sort(key=lambda g: -p_of.get(g[0], 0.0))
        (m1, r1, g1), (m2, r2, g2) = good
        if g1["tool_uses"] or g2["tool_uses"]:
            pick = next(g for g in good if g[2]["tool_uses"])   # a tool call can't be merged; take the better-placed one
            info["synthesis"] = "tool_turn"
            return pick[1], pick[0]
        sim = verify.similarity(g1["text"], g2["text"])
        info["agreement"] = round(sim, 2)
        if sim >= syn["agree_above"]:
            info["synthesis"] = "agree"
            return r1, m1
        arbiter = arms.get(syn.get("arbiter") or "")
        merger = arbiter["model"] if arbiter and sim < syn["agree_above"] / 2 else arms[syn["merger"]]["model"]
        o2 = synth_request(obj, [g1["text"], g2["text"]])
        try:
            r3 = self._fetch(json.dumps(o2).encode(), o2, merger)
        except RouteError:
            r3 = None
        if r3 is not None and r3.status == 200 and not [i for i in verify.check(verify.parse(r3.body, r3.ctype), o2)
                                                        if i[1] == "hard"]:
            info["synthesis"] = "merged"
            info["merger"] = merger
            return r3, merger
        info["synthesis"] = "merge_failed"
        return r1, m1


def build_server(cfg):
    Handler.cfg, Handler.key = cfg, read_key()
    Handler.auto = auto_mod.Auto(cfg, DIR, BACKENDS) if cfg.get("auto") else None
    if Handler.auto:
        Handler.auto.warm_up()
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer(("127.0.0.1", cfg["port"]), Handler)


def healthy(port):
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        c.request("GET", "/health")
        return c.getresponse().status == 200
    except OSError:
        return False


def cmd_init(cfg, quiet=False):
    os.makedirs(DIR, mode=0o700, exist_ok=True)
    kp = os.path.join(DIR, "key")
    if not os.path.exists(kp):
        fd = os.open(kp, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    settings = {"env": {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % cfg["port"],
        "ANTHROPIC_CUSTOM_HEADERS": "x-router-key: " + read_key(),
        "ENABLE_TOOL_SEARCH": str(cfg["tool_search"]),
    }}
    sp = os.path.join(DIR, "settings.json")
    fd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(settings, f, indent=2)
    if not quiet:
        print("wrote", kp, "and", sp)
        print("launch Claude Code with:  backend-switchboard run   (then /model <name>[@argo|@mag|@alcf|@asksage])")


def cmd_ensure(cfg):
    if healthy(cfg["port"]):
        return
    log = open(os.path.join(DIR, "router.log"), "ab")
    p = subprocess.Popen([sys.executable, "-m", "backend_switchboard", "serve"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    with open(os.path.join(DIR, "router.pid"), "w") as f:
        f.write(str(p.pid))
    for _ in range(20):
        if healthy(cfg["port"]):
            return
        time.sleep(0.25)
    sys.exit("backend-switchboard failed to start; see %s/router.log" % DIR)


def cmd_run(cfg, args):
    cmd_init(cfg, quiet=True)  # keeps settings.json in sync with config.json (port)
    cmd_ensure(cfg)
    if shutil.which("claude") is None:
        sys.exit("claude not found on PATH; install Claude Code first")
    os.execvp("claude", ["claude", "--settings", os.path.join(DIR, "settings.json")] + args)


def cmd_stop():
    try:
        pid = int(open(os.path.join(DIR, "router.pid")).read())
        os.kill(pid, 15)
        print("stopped", pid)
    except (OSError, ValueError):
        print("not running")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run":  # everything after "run" belongs to claude
        return cmd_run(load_config(), sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "serve", "ensure", "stop", "decide"],
                    help="run [claude args...] is also accepted: starts the router if needed, then launches claude")
    ap.add_argument("text", nargs="?", help="decide: the prompt to classify (default: read stdin)")
    ap.add_argument("--tools", nargs="*", default=[], metavar="NAME",
                    help="decide: pretend the request carries these tools, e.g. --tools Bash Edit")
    args = ap.parse_args()
    cmd = args.command
    cfg = load_config()
    if cmd == "decide":
        a = auto_mod.Auto(cfg, DIR, BACKENDS)
        text = args.text if args.text is not None else sys.stdin.read()
        p = a.plan({"messages": [{"role": "user", "content": text}], "tools": [{"name": n} for n in args.tools]})
        print(json.dumps({"decision": p.decision, "candidates": p.candidates} if p else {"error": "no eligible backend"},
                         indent=2))
    elif cmd == "init":
        cmd_init(cfg)
    elif cmd == "serve":
        srv = build_server(cfg)
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    elif cmd == "ensure":
        cmd_ensure(cfg)
    else:
        cmd_stop()


if __name__ == "__main__":
    main()
