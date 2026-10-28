#!/usr/bin/env python3
"""Standalone check: python3.11 tests/selftest.py  (no network, no credentials, mock upstreams)."""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

tmp = tempfile.mkdtemp()
os.environ["BACKEND_SWITCHBOARD_HOME"] = tmp
os.environ["AMSC_TOKEN"] = "MAGTOKEN"
os.environ["ALCF_INFERENCE_TOKEN"] = "ALCFTOKEN"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import backend_switchboard._router as cr  # noqa: E402

SEEN = {}


def mock(name):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.loads(raw)
            SEEN[name] = {"path": self.path, "h": {k.lower(): v for k, v in self.headers.items()}, "raw": raw, "body": body}
            if body["model"] == "boom":
                out = b'{"type":"error","error":{"type":"api_error","message":"overloaded"}}'
                self.send_response(503)
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                return self.wfile.write(out)
            if body["model"] == "revoked":
                out = b"IMPORTANT USAGE NOTICE FROM ARGO ... ACCESS REVOKED"
                self.send_response(200)
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                return self.wfile.write(out)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: a\ndata: 1\n\n")
            self.wfile.flush()
            time.sleep(0.6 if body["model"] == "stream" else 0)
            self.wfile.write(b"event: b\ndata: 2\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return "http://127.0.0.1:%d" % srv.server_address[1]


urls = {n: mock(n) for n in ("anthropic", "mag", "argo", "alcf")}
argo_settings = os.path.join(tmp, "argo_settings.json")
json.dump({"apiKeyHelper": "echo 'ARGOTOKEN'", "env": {"ANTHROPIC_BASE_URL": urls["argo"] + "/argoapi"}},
          open(argo_settings, "w"))
D1 = {"features": {"expert": 0.0, "difficulty": 1.0}, "fail": False, "calls": 0, "texts": []}


class D1Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        D1["calls"] += 1
        D1["texts"].append(req["text"])
        if D1["fail"]:
            self.send_response(500)
            self.send_header("content-length", "0")
            return self.end_headers()
        out = json.dumps({"features": D1["features"], "ms": 1.0}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


d1_srv = ThreadingHTTPServer(("127.0.0.1", 0), D1Handler)
threading.Thread(target=d1_srv.serve_forever, daemon=True).start()
TIERS = {"small": ["claude-small@mag"], "medium": ["claude-mid@argo"], "large": ["claude-big@alcf"]}
json.dump({"port": 0, "anthropic_url": urls["anthropic"], "mag_url": urls["mag"],
           "argo_settings": argo_settings, "alcf_url": urls["alcf"],
           "auto": {"d1_url": "http://127.0.0.1:%d" % d1_srv.server_address[1], "tiers": TIERS}},
          open(os.path.join(tmp, "config.json"), "w"))
cr.cmd_init(cr.load_config())
KEY = cr.read_key()
srv = cr.build_server(cr.load_config())
threading.Thread(target=srv.serve_forever, daemon=True).start()
PORT = srv.server_address[1]
fails = []


def check(label, ok, detail=""):
    print("%s  %s %s" % ("ok  " if ok else "FAIL", label, "" if ok else detail))
    if not ok:
        fails.append(label)


def post(model, headers=None, key=KEY, extra=None):
    h = {"content-type": "application/json", "authorization": "Bearer CLIENT-OAUTH", "anthropic-beta": "b1,b2"}
    if key:
        h["x-router-key"] = key
    h.update(headers or {})
    raw = json.dumps(dict({"model": model, "max_tokens": 5, "messages": []}, **(extra or {}))).encode()
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    t0 = time.time()
    c.request("POST", "/v1/messages", raw, h)
    r = c.getresponse()
    first = r.read1(100)
    t_first = time.time() - t0
    return r.status, first + r.read(), t_first, raw


s, _, _, _ = post("claude-sonnet", key=None)
check("rejects missing x-router-key", s == 401)

s, _, _, raw = post("claude-opus-5-5")
a = SEEN["anthropic"]
check("no tag -> anthropic, status 200", s == 200)
check("anthropic: body bytes identical", a["raw"] == raw)
check("anthropic: client auth + beta pass through", a["h"].get("authorization") == "Bearer CLIENT-OAUTH" and a["h"].get("anthropic-beta") == "b1,b2")
check("anthropic: router key not leaked upstream", "x-router-key" not in a["h"])

post("claude-opus-5-5@google@mag")
m = SEEN["mag"]
check("@mag: tag stripped, MAG suffix kept", m["body"]["model"] == "claude-opus-5-5@google")
check("@mag: client auth replaced by Bearer MAG token", m["h"].get("authorization") == "Bearer MAGTOKEN" and "x-api-key" not in m["h"])
check("@mag: anthropic-beta still forwarded", m["h"].get("anthropic-beta") == "b1,b2")

post("claude-sonnet-5-5@argo")
g = SEEN["argo"]
check("@argo: path under /argoapi, token from apiKeyHelper (shlex)", g["path"] == "/argoapi/v1/messages" and g["h"].get("x-api-key") == "ARGOTOKEN")
check("@argo: client OAuth not forwarded", "authorization" not in g["h"])

post("beam@alcf")
l = SEEN["alcf"]
check("@alcf: alias beam -> beam-1.0-bf16, Bearer ALCF token, client auth dropped",
      l["body"]["model"] == "beam-1.0-bf16" and l["h"].get("authorization") == "Bearer ALCFTOKEN" and "x-api-key" not in l["h"])
check("@alcf: path is /v1/messages (native, no translation)", l["path"] == "/v1/messages")
post("some-new-model@alcf")
check("@alcf: unknown names pass through", SEEN["alcf"]["body"]["model"] == "some-new-model")

ts = {"tools": [{"name": "Read", "input_schema": {}}, {"name": "DeferredToolPlaceholder", "defer_loading": True, "input_schema": {}}],
      "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                                 "content": [{"type": "tool_reference", "tool_name": "WebFetch"}]}]}]}
_, _, _, raw_ts = post("claude-opus-5-5", extra=ts)
check("tool search: subscription route left byte-identical", SEEN["anthropic"]["raw"] == raw_ts)
for tag in ("mag", "argo", "alcf"):
    post("claude-x@" + tag, extra=ts)
    b = SEEN[tag]["body"]
    check("tool search flattened for @" + tag,
          [t["name"] for t in b["tools"]] == ["Read"]
          and b["messages"][0]["content"][0]["content"][0]["type"] == "text"
          and "WebFetch" in b["messages"][0]["content"][0]["content"][0]["text"])

ext = {"messages": [{"role": "user", "output_config": {"x": 1}, "content": [{"type": "text", "text": "hi"}, {"type": "tool_addition", "tools": ["A"]}]}]}
_, _, _, raw_ext = post("claude-opus-5-5", extra=ext)
check("extensions: subscription route left byte-identical", SEEN["anthropic"]["raw"] == raw_ext)
for tag in ("mag", "argo", "alcf"):
    post("claude-x@" + tag, extra=ext)
    m0 = SEEN[tag]["body"]["messages"][0]
    check("extensions stripped for @" + tag, set(m0) == {"role", "content"} and [b["type"] for b in m0["content"]] == ["text"])

s, body, _, _ = post("claude-opus-5-5@google")
check("ambiguous '@google' -> 400 with guidance", s == 400 and b"@mag" in body)

s, body, t_first, _ = post("stream")
check("streaming: first event arrives before upstream finishes", s == 200 and t_first < 0.45 and b"event: b" in body, "t_first=%.2f" % t_first)

c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
c.request("POST", "/v1/messages", json.dumps({"model": "revoked@argo", "messages": []}), {"x-router-key": KEY})
r = c.getresponse()
check("argo soft error (HTTP 200 + ACCESS REVOKED) -> 429", r.status == 429)

# --- auto routing -----------------------------------------------------------------------------------------------------
AUTO_LOG = os.path.join(tmp, "auto.jsonl")
EASY = {"expert": 0.0, "difficulty": 1.0}      # hardness -1.92 -> small
MID = {"expert": 0.0, "difficulty": 1.7}       # hardness -0.32 -> medium
HARD = {"expert": 1.0, "difficulty": 4.0}      # hardness  8.7  -> large


def ask(text, prior=None, features=None, model="auto"):
    """One `auto` request whose latest typed turn is `text`; returns the HTTP status."""
    D1["features"] = features or EASY
    msgs = list(prior or []) + [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder>ignore me</system-reminder>"}, {"type": "text", "text": text}]}]
    return post(model, extra={"messages": msgs})[0]


def log_lines(n, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        try:
            lines = [json.loads(l) for l in open(AUTO_LOG)]
        except FileNotFoundError:
            lines = []
        if len(lines) >= n:
            return lines
        time.sleep(0.05)
    return lines


check("auto: d1 is warmed up at startup", "warm up" in D1["texts"])
s = ask("what is 2+2", features=EASY)
check("auto: easy -> small tier backend, tag stripped", s == 200 and SEEN["mag"]["body"]["model"] == "claude-small")
check("auto: d1 sees the typed turn only (reminders dropped)", D1["texts"][-1] == "what is 2+2")
s = ask("write a scheduler", features=MID)
check("auto: medium -> argo", s == 200 and SEEN["argo"]["body"]["model"] == "claude-mid" and SEEN["argo"]["h"].get("x-api-key") == "ARGOTOKEN")
s = ask("prove the theorem", features=HARD)
check("auto: hard -> alcf with Bearer", s == 200 and SEEN["alcf"]["body"]["model"] == "claude-big" and SEEN["alcf"]["h"].get("authorization") == "Bearer ALCFTOKEN")

calls = D1["calls"]
ask("prove the theorem", features=EASY)  # same turn again (a tool-loop request): cached decision, features ignored
check("auto: repeated turn reuses the cached decision (one d1 call)", D1["calls"] == calls
      and SEEN["alcf"]["body"]["model"] == "claude-big")

prior = [{"role": "user", "content": "refactor the lock manager across modules"}, {"role": "assistant", "content": "ok"}]
ask("yes", prior=prior)
check("auto: a very short turn is judged together with the previous one",
      D1["texts"][-1] == "refactor the lock manager across modules\n\nyes")

D1["fail"] = True
s = ask("a fresh prompt while d1 is down")
check("auto: d1 down -> heuristic decides, request still served", s == 200 and SEEN["mag"]["body"]["model"] == "claude-small")
check("auto: heuristic source is logged", log_lines(1)[-1]["decision"]["source"].startswith("heuristic"))
D1["fail"] = False
cr.Handler.auto.d1.until.clear()

cr.Handler.auto.c["tiers"] = {"small": ["boom@mag", "claude-small@argo"], "medium": TIERS["medium"], "large": TIERS["large"]}
s = ask("fallback prompt one")
check("auto: 503 from the first candidate -> next candidate answers", s == 200 and SEEN["argo"]["body"]["model"] == "claude-small")
check("auto: failed attempt is recorded", log_lines(1)[-1]["attempts"] == [["boom@mag", 503]]
      and log_lines(1)[-1]["chosen"] == "claude-small@argo")
ask("fallback prompt two")
ask("fallback prompt three")
check("auto: breaker moves a failing backend to the back", cr.Handler.auto.candidates("small")[0] == "claude-small@argo")

cr.Handler.auto.c["tiers"] = {"small": ["boom@mag"]}
s = ask("only candidate fails")
check("auto: when every candidate fails the last error reaches the client", s == 503)
cr.Handler.auto.c["tiers"], cr.Handler.auto.c["enabled_arms"] = {}, []
s = ask("nothing eligible")
check("auto: no tiers and no arms -> 400 with guidance", s == 400)
cr.Handler.auto.c["enabled_arms"] = None

# arms mode (no tiers): the registry is ranked per request
D1["features"] = dict(EASY, **{"domain=chat": 1.0})
s = ask("hello there", features=dict(EASY, **{"domain=chat": 1.0}))
check("arms: easy chat goes to a free ALCF arm", s == 200 and SEEN["alcf"]["body"]["model"] in ("beam-1.0-bf16", "muse-spark-bf16"))
ranked = cr.Handler.auto.arms({"messages": []}, "x")[0]["ranked"]
check("arms: ranking is logged with p and cost", ranked and {"arm", "p", "cost"} <= set(ranked[0]))
D1["features"] = dict(HARD, **{"domain=science": 1.0})
d, rk = cr.Handler.auto.arms({"messages": []}, "hard science question", 4000)
check("arms: hard science is not left to a free arm", rk and rk[0]["arm"] in ("opus", "sonnet"))
img = {"messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}
check("arms: images only go to vision-capable arms", all(r["arm"] != "beam" for r in cr.Handler.auto.arms(img, "x")[1]))
check("arms: oversized prompt excludes small-context arms",
      all(r["arm"] != "beam" for r in cr.Handler.auto.arms({"messages": []}, "x", 4000000)[1]))
cr.Handler.auto.c["arms"] = {"muse": None, "beam": None}
check("arms: config can remove an arm", all(r["arm"] not in ("beam", "muse") for r in cr.Handler.auto.arms({"messages": []}, "x")[1]))
cr.Handler.auto.c["arms"] = {"mine": {"model": "claude-x@mag", "pool": "alcf", "price": [0, 0], "skills": [0.9] * 6}}
check("arms: config can add an arm", any(r["arm"] == "mine" for r in cr.Handler.auto.arms({"messages": []}, "x")[1]))
cr.Handler.auto.c["arms"] = {}
cr.Handler.auto.c["tiers"] = TIERS

text = "ZZ-secret-prompt-text"
ask(text)
raw_log = open(AUTO_LOG).read()
check("auto: telemetry holds a hash, never the prompt", text not in raw_log and "prompt_sha1" in raw_log)
check("auto: telemetry file is private", oct(os.stat(AUTO_LOG).st_mode & 0o777) == "0o600")

before = len(log_lines(0))
cr.Handler.auto.c["shadow"] = True
s, _, _, raw_sh = post("claude-opus-5-5", extra={"messages": [{"role": "user", "content": "shadowed question"}]})
sh = [l for l in log_lines(before + 1) if l.get("shadow")]
check("auto shadow: explicit model untouched, decision logged", s == 200 and SEEN["anthropic"]["raw"] == raw_sh
      and sh and sh[-1]["requested"] == "claude-opus-5-5" and sh[-1]["would_use"])
cr.Handler.auto.c["shadow"] = False

print("\n%d failure(s)" % len(fails))
sys.stdout.flush()
sys.stderr.flush()
# Python 3.8 aborts at shutdown if a daemon handler thread is still logging to stderr, so skip normal teardown.
os._exit(1 if fails else 0)
