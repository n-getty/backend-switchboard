#!/usr/bin/env python3
"""backend-switchboard: one local Anthropic-API endpoint for Claude Code, routed by /model suffix.

  claude-opus-5-5              -> Anthropic (your subscription; client headers pass through untouched)
  claude-opus-5-5@google@mag   -> AMSC MAG   (router tag @mag is stripped; "@google" is MAG's own suffix)
  claude-opus-5-5@argo         -> Argo via argo-shim
  beam@alcf / muse@alcf        -> ALCF Minerva (serves the Anthropic Messages API natively; Bearer token)
  gpt-5.4@asksage              -> AskSage /query (translated in-process; needs ASKSAGE_API_KEY)
  claude-opus-5-5@copilot      -> GitHub Copilot (Claude: native; gpt-*/grok-*: /responses, translated; uses `gh auth token`)

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

from . import _asksage as asksage
from . import _copilot as copilot

HOME = os.path.expanduser("~")
DIR = os.environ.get("BACKEND_SWITCHBOARD_HOME", os.path.join(HOME, ".backend-switchboard"))
BACKENDS = ("anthropic", "argo", "mag", "alcf", "asksage", "copilot")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
SOFT_ERRORS = (b"ACCESS REVOKED", b"IMPORTANT USAGE NOTICE")

DEFAULTS = {
    "port": 8790,
    "tool_search": "true",
    "anthropic_url": "https://api.anthropic.com",
    "mag_url": "https://mag.genesis.americansciencecloud.org",
    "argo_settings": os.path.join(HOME, ".claude", "settings.json"),
    "argo_url": None,    # e.g. http://127.0.0.1:20098/argoapi (else $ARGO_SHIM_URL, else read from argo_settings)
    "argo_token": None,  # else argo_settings' apiKeyHelper, else $ARGO_SHIM_TOKEN
    "alcf_url": "https://inference-api.alcf.anl.gov/resource_server/minerva/api",
    "asksage_url": "https://api.asksage.anl.gov/server",
    "asksage_cert": None,  # extra CA bundle (ANL's AskSage cert), e.g. ~/.config/asksage_proxy/asksage_anl_gov.pem
    "copilot_url": "https://api.githubcopilot.com",
    # /model picker rows: "name@tag" or {"model": ..., "label": ..., "description": ...}. Setting "models" in config.json
    # replaces this list ([] shows none).
    "models": [
        {"model": "claude-sonnet-5-5@argo", "label": "Sonnet 5.5 (Argo)"},
        {"model": "claude-opus-5-5@google@mag", "label": "Opus 5.5 (MAG)"},
        {"model": "beam@alcf", "label": "Beam (ALCF)"},
        {"model": "muse@alcf", "label": "Muse (ALCF)"},
        {"model": "google-claude-opus-5@asksage", "label": "Claude Opus 5 (AskSage)"},
        {"model": "claude-opus-5-5@copilot", "label": "Opus 5.5 (Copilot)"},
    ],
    # an alias is a model name, or {"model": ..., "url": ...} when the model lives on another ALCF cluster
    "aliases": {"alcf": {"beam": {"model": "beam-1.0-fp8", "url": "https://inference-api.alcf.anl.gov/resource_server/tara/api"},
                         "muse": "muse-spark-bf16"}},
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
        st = os.stat(path)
        m = (st.st_mtime_ns, st.st_size)
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


def _config_argo(path):
    d = json.load(open(path))
    return d.get("argo_token"), d.get("argo_url")


def argo_creds(cfg):
    """(token, base URL) for argo-shim. Explicit config wins; then the settings file (re-read on change, so a rotated
    token is picked up); then the environment. Nothing needs to live in Claude Code's own settings."""
    # config.json is re-read on change, so `migrate` (or an edit) works without restarting a running router
    live_tok, live_url = cached_file(os.path.join(DIR, "config.json"), _config_argo) or (None, None)
    tok = (cfg.get("argo_token") or live_tok or cached_file(cfg["argo_settings"], _argo_token)
           or os.environ.get("ARGO_SHIM_TOKEN"))
    url = (cfg.get("argo_url") or live_url or os.environ.get("ARGO_SHIM_URL")
           or cached_file(cfg["argo_settings"], _argo_base))
    return tok, url


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
        tok, url = argo_creds(cfg)
        if not tok or not url:
            raise RouteError(502, "no argo-shim token/URL: set argo_url and argo_token in config.json (or "
                                  "ARGO_SHIM_URL / ARGO_SHIM_TOKEN), or point argo_settings at a file with them")
        return url, {"x-api-key": tok}, True, base, tag
    if tag == "asksage":
        return cfg["asksage_url"], {}, True, base, tag
    if tag == "copilot":
        tok = copilot.token()
        if not tok:
            raise RouteError(401, "no GitHub token; run in your own terminal: gh auth login (or set COPILOT_GITHUB_TOKEN)")
        return (cfg["copilot_url"], dict(copilot.HEADERS, authorization="Bearer " + tok), True,
                cfg["aliases"].get("copilot", {}).get(base, copilot.upstream_name(base)), tag)
    tok = alcf_token()
    if not tok:
        raise RouteError(401, "no ALCF token; run in your own terminal: alcf-tokens login inference")
    alias = cfg["aliases"]["alcf"].get(base, base)
    url, model = (alias.get("url", cfg["alcf_url"]), alias["model"]) if isinstance(alias, dict) else (cfg["alcf_url"], alias)
    return url, {"authorization": "Bearer " + tok}, True, model, tag


BLOCK_TYPES = {"text", "image", "document", "tool_use", "tool_result", "thinking", "redacted_thinking",
               "server_tool_use", "web_search_tool_result", "search_result", "container_upload"}


def strip_client_extensions(obj):
    """Drop Claude Code's Anthropic-only extras that Vertex-backed Argo rejects with a 400: message-level keys other
    than role/content (e.g. output_config) and content blocks of unknown type (e.g. tool_addition).
    Returns True if anything changed."""
    changed = False
    msgs = obj.get("messages", [])
    for msg in msgs:
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
    # A message whose only blocks were stripped (e.g. a mid-conversation role "system" message carrying a block type
    # Argo does not know) would reach Vertex empty and be rejected: "messages.N: system content must contain at least
    # one block". Drop such messages; neighbours of the same role are merged by the API.
    live = [m for m in msgs if not (isinstance(m, dict) and m.get("content") == [])]
    if len(live) != len(msgs):
        obj["messages"] = live
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
            if backend == "asksage":
                status = asksage.handle(self, self.cfg, obj)
            else:
                kind = copilot.endpoint(self.cfg, up_model, extra["authorization"][7:]) if backend == "copilot" else None
                if kind and kind != "messages":
                    status = copilot.handle(self, self.cfg, obj, extra["authorization"][7:], up_model, kind)
                else:
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
    rows = []
    for m in cfg.get("models") or []:
        row = {"model": m} if isinstance(m, str) else dict(m)
        if row.get("model"):
            row.setdefault("label", row["model"])
            rows.append(row)
    if rows:
        settings["modelPicker"] = {"options": rows}
    sp = os.path.join(DIR, "settings.json")
    fd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(settings, f, indent=2)
    if not quiet:
        print("wrote", kp, "and", sp)
        print("launch Claude Code with:  backend-switchboard run   (then /model <name>[@argo|@mag|@alcf|@asksage|@copilot])")


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


def settings_conflicts(cfg):
    """Claude Code settings that make the plain-Anthropic (subscription) route fail: left over from an Argo-only setup."""
    cdir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(HOME, ".claude")
    try:
        with open(os.path.join(cdir, "settings.json")) as f:
            user = json.load(f)
    except (OSError, ValueError):
        user = {}
    env = dict(user.get("env", {}))
    env.update({k: v for k, v in os.environ.items() if k == "CLAUDE_CODE_SKIP_ANTHROPIC_AUTH"})
    found = []
    if str(env.get("CLAUDE_CODE_SKIP_ANTHROPIC_AUTH", "")).lower() in ("1", "true", "yes"):
        found.append("CLAUDE_CODE_SKIP_ANTHROPIC_AUTH is set, so Claude Code skips your Anthropic login")
    if env.get("ANTHROPIC_BASE_URL"):
        found.append("ANTHROPIC_BASE_URL in %s sends plain `claude` straight to %s, not through the switchboard"
                     % (os.path.join(cdir, "settings.json"), env["ANTHROPIC_BASE_URL"]))
    helper = user.get("apiKeyHelper")
    words = shlex.split(helper) if isinstance(helper, str) and helper.strip() else []
    argo_tok = argo_creds(cfg)[0]
    if words and argo_tok and words[-1] == argo_tok:
        found.append("apiKeyHelper in %s sends the argo-shim key on every request" % os.path.join(cdir, "settings.json"))
    return found


def _write_json_600(path, data):
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def cmd_migrate(cfg, yes=False):
    """Move an Argo-only Claude Code setup behind the switchboard: shim URL/token into config.json, and remove
    ANTHROPIC_BASE_URL, apiKeyHelper and CLAUDE_CODE_SKIP_ANTHROPIC_AUTH from Claude Code's settings.json."""
    cdir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(HOME, ".claude")
    sp = os.path.join(cdir, "settings.json")
    try:
        with open(sp) as f:
            user = json.load(f)
    except (OSError, ValueError):
        user = {}
    tok, url = argo_creds(cfg)  # read before anything is removed
    env = user.get("env", {})
    helper = user.get("apiKeyHelper")
    words = shlex.split(helper) if isinstance(helper, str) and helper.strip() else []
    remove = []
    if str(env.get("CLAUDE_CODE_SKIP_ANTHROPIC_AUTH", "")).lower() in ("1", "true", "yes"):
        remove.append("env.CLAUDE_CODE_SKIP_ANTHROPIC_AUTH")
    if env.get("ANTHROPIC_BASE_URL") and urlsplit(env["ANTHROPIC_BASE_URL"]).hostname in ("127.0.0.1", "localhost"):
        remove.append("env.ANTHROPIC_BASE_URL")
    if words and tok and words[-1] == tok:
        remove.append("apiKeyHelper")
    shell_skip = str(os.environ.get("CLAUDE_CODE_SKIP_ANTHROPIC_AUTH", "")).lower() in ("1", "true", "yes")
    if not remove and not shell_skip:
        print("nothing to migrate: %s has no Argo-only settings" % sp)
        return
    print("Claude Code config dir: %s" % cdir)
    if remove:
        print("will remove from %s: %s" % (sp, ", ".join(remove)))
        print("will save the Argo URL and token to %s (mode 600)" % os.path.join(DIR, "config.json"))
        print("will back up settings.json to %s.bak-switchboard (mode 600; it contains the token, delete it when happy)" % sp)
    print("!! Close every running Claude Code session that uses this setup first: they re-read settings.json and the Argo\n"
          "   ones will fail midstream with 'Login expired'.")
    if remove and not yes and (input("Continue? [y/N] ").strip().lower() != "y"):
        return print("aborted, nothing changed")
    if remove:
        if tok and url:
            os.makedirs(DIR, mode=0o700, exist_ok=True)
            cp = os.path.join(DIR, "config.json")
            try:
                with open(cp) as f:
                    conf = json.load(f)
            except (OSError, ValueError):
                conf = {}
            conf.setdefault("argo_url", url)
            conf.setdefault("argo_token", tok)
            _write_json_600(cp, conf)
        else:
            print("warning: no Argo token/URL found to save; @argo will need argo_url/argo_token in config.json")
        _write_json_600(sp + ".bak-switchboard", user)
        user.pop("apiKeyHelper", None) if "apiKeyHelper" in remove else None
        for k in ("CLAUDE_CODE_SKIP_ANTHROPIC_AUTH", "ANTHROPIC_BASE_URL"):
            if "env." + k in remove:
                env.pop(k, None)
        if "env" in user and not user["env"]:
            del user["env"]
        _write_json_600(sp, user)
        cmd_stop()  # a running router would otherwise keep its old view of the config
        print("done: settings.json cleaned. Start Claude with `backend-switchboard run` and /login once.")
    if shell_skip:
        print("!! CLAUDE_CODE_SKIP_ANTHROPIC_AUTH is also exported in your shell, which this tool cannot change.\n"
              "   Run `unset CLAUDE_CODE_SKIP_ANTHROPIC_AUTH` and delete the export from ~/.bashrc / ~/.zshrc / ~/.profile\n"
              "   (find it with: grep -n CLAUDE_CODE_SKIP_ANTHROPIC_AUTH ~/.bashrc ~/.zshrc ~/.profile ~/.bash_profile).")


def cmd_run(cfg, args):
    cmd_init(cfg, quiet=True)  # keeps settings.json in sync with config.json (port)
    problems = settings_conflicts(cfg)
    if problems:
        sys.stderr.write("backend-switchboard: warning: untagged models (your subscription) will fail:\n"
                         + "".join("  - %s\n" % p for p in problems)
                         + "  Fix: close running Claude Code sessions, then run `backend-switchboard migrate`.\n")
    cmd_ensure(cfg)
    if shutil.which("claude") is None:
        sys.exit("claude not found on PATH; install Claude Code first")
    os.execvp("claude", ["claude", "--settings", os.path.join(DIR, "settings.json")] + args)


def cmd_doctor(cfg):
    """Print what is and is not set up, per backend, so first-time setup problems are visible before a session breaks."""
    def row(ok, name, msg):
        print("%-4s %-10s %s" % ("ok" if ok else "FIX", name, msg))
    cdir = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(HOME, ".claude")
    print("Claude Code config dir: %s" % cdir)
    row(not tls_broken(), "tls", "HTTPS context works" + (" (OPENSSL_CONF=%s)" % os.environ["OPENSSL_CONF"]
                                                           if "OPENSSL_CONF" in os.environ else "")
        if not tls_broken() else "Python's OpenSSL rejects the system openssl.cnf; restart via backend-switchboard "
                                 "(sets OPENSSL_CONF=/dev/null) or export OPENSSL_CONF=/dev/null")
    row(healthy(cfg["port"]), "router", "listening on 127.0.0.1:%d" % cfg["port"] if healthy(cfg["port"])
        else "not running (`backend-switchboard ensure`)")
    problems = settings_conflicts(cfg)
    row(not problems, "settings", "no Argo-only leftovers in Claude Code settings" if not problems
        else "; ".join(problems) + "  -> close running Claude sessions, then run: backend-switchboard migrate")
    tok, url = argo_creds(cfg)
    row(bool(tok and url), "argo", ("token and %s" % url) if tok and url else "set argo_url/argo_token in config.json (optional)")
    row(bool(mag_token()), "mag", "AMSC_TOKEN set" if mag_token() else "export AMSC_TOKEN (optional)")
    row(bool(os.environ.get("ALCF_INFERENCE_TOKEN") or shutil.which("alcf-tokens")), "alcf",
        "token tools found" if shutil.which("alcf-tokens") or os.environ.get("ALCF_INFERENCE_TOKEN")
        else "uv tool install alcf-ai alcf-tokens; alcf-tokens login inference (optional)")
    ask = os.environ.get("ASKSAGE_API_KEY") or os.environ.get("ASKSAGE_PROXY_KEY")
    row(bool(ask), "asksage", "API key set" if ask
        else "export ASKSAGE_API_KEY in the shell that starts the router (optional)")
    gh = any(os.environ.get(k) for k in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")) or shutil.which("gh")
    row(bool(gh), "copilot", "gh or token env found" if gh else "gh auth login (optional)")
    print("anthropic: no setup here; run /login inside Claude Code once. Optional rows are only needed for the backends you use.")


def cmd_stop():
    if not os.path.exists(os.path.join(DIR, "router.pid")):
        return print("not running")
    try:
        pid = int(open(os.path.join(DIR, "router.pid")).read())
        os.kill(pid, 15)
        print("stopped", pid)
    except (OSError, ValueError):
        print("not running")


def tls_broken():
    """True if Python cannot build a default TLS context, probed in a fresh process (OpenSSL reports a bad config only
    on the first use in a process). Seen when the system openssl.cnf has settings the bundled OpenSSL cannot initialize
    (e.g. `seed=JITTER` with OpenSSL 3.5): every HTTPS route then fails with `SSLError ... (_ssl.c:3036)`."""
    probe = "import ssl; ssl.create_default_context()"
    return subprocess.call([sys.executable, "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0


def main():
    # OpenSSL reads its config when `ssl` is imported, so the only fix is to restart with a neutral config.
    if "OPENSSL_CONF" not in os.environ and tls_broken():
        os.environ["OPENSSL_CONF"] = os.devnull
        os.execv(sys.executable, [sys.executable, "-m", "backend_switchboard"] + sys.argv[1:])
    if len(sys.argv) > 1 and sys.argv[1] == "run":  # everything after "run" belongs to claude
        return cmd_run(load_config(), sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "serve", "ensure", "stop", "doctor", "migrate"],
                    help="run [claude args...] is also accepted: starts the router if needed, then launches claude")
    ap.add_argument("--yes", "-y", action="store_true", help="migrate: do not ask for confirmation")
    ns = ap.parse_args()
    cmd = ns.command
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
    elif cmd == "doctor":
        cmd_doctor(cfg)
    elif cmd == "migrate":
        cmd_migrate(cfg, ns.yes)
    else:
        cmd_stop()


if __name__ == "__main__":
    main()
