#!/usr/bin/env python3
"""backend-switchboard: one local Anthropic-API endpoint for Claude Code, routed by /model suffix.

  claude-opus-5-5              -> Anthropic (your subscription; client headers pass through untouched)
  claude-opus-5-5@google@mag   -> AMSC MAG   (router tag @mag is stripped; "@google" is MAG's own suffix)
  claude-opus-5-5@argo         -> Argo via argo-shim
  beam@alcf / muse@alcf        -> ALCF Minerva (serves the Anthropic Messages API natively; Bearer token)

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOME = os.path.expanduser("~")
DIR = os.environ.get("BACKEND_SWITCHBOARD_HOME", os.path.join(HOME, ".backend-switchboard"))
BACKENDS = ("anthropic", "argo", "mag", "alcf")
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


class Handler(BaseHTTPRequestHandler):
    cfg = key = None
    server_version = "backend-switchboard"

    def log_message(self, *a):
        pass

    def _err(self, status, message):
        body = json.dumps({"type": "error", "error": {"type": "invalid_request_error" if status < 500 else "api_error",
                                                       "message": message}}).encode()
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
        model, backend = "-", "?"
        try:
            body = raw
            if raw and self.path.split("?")[0].startswith("/v1/messages"):
                try:
                    obj = json.loads(raw)
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and isinstance(obj.get("model"), str):
                    model = obj["model"]
            url, extra, drop_auth, up_model, backend = resolve(self.cfg, model) \
                if model != "-" else (self.cfg["anthropic_url"], {}, False, None, "anthropic")
            if model != "-":
                dirty = up_model != model
                obj["model"] = up_model
                if backend != "anthropic":
                    dirty = flatten_tool_search(obj) or dirty
                    dirty = strip_client_extensions(obj) or dirty
                if dirty:
                    body = json.dumps(obj).encode()
            status = self._forward(url, extra, drop_auth, body, backend)
        except RouteError as e:
            status = e.status
            self._err(e.status, e.message)
        except Exception as e:
            status = 502 if isinstance(e, (OSError, http.client.HTTPException)) else 500
            sys.stderr.write("router error: %r\n" % (e,))
            try:
                self._err(status, "router/upstream error: %s" % e)
            except OSError:
                pass
        sys.stderr.write("%s %-40s -> %s %s %.2fs\n" % (time.strftime("%H:%M:%S"), model[:40],
                         backend, status, time.time() - t0))
        sys.stderr.flush()

    def _forward(self, url, extra, drop_auth, body, backend):
        u = urlsplit(url)
        headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in HOP or lk == "x-router-key":
                continue
            if drop_auth and lk in ("authorization", "x-api-key"):
                continue
            if backend == "argo" and lk == "accept-encoding":
                continue
            headers[lk] = v
        headers.update(extra)
        if u.scheme == "https":
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=900, context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=900)
        conn.request(self.command, u.path.rstrip("/") + self.path, body or None, headers)
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


def build_server(cfg):
    Handler.cfg, Handler.key = cfg, read_key()
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
        print("launch Claude Code with:  backend-switchboard run   (then /model <name>[@argo|@mag|@alcf])")


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
    ap.add_argument("command", choices=["init", "serve", "ensure", "stop"],
                    help="run [claude args...] is also accepted: starts the router if needed, then launches claude")
    cmd = ap.parse_args().command
    cfg = load_config()
    if cmd == "init":
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
