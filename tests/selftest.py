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
json.dump({"port": 0, "anthropic_url": urls["anthropic"], "mag_url": urls["mag"],
           "argo_settings": argo_settings, "alcf_url": urls["alcf"]},
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

print("\n%d failure(s)" % len(fails))
sys.stdout.flush()
sys.stderr.flush()
# Python 3.8 aborts at shutdown if a daemon handler thread is still logging to stderr, so skip normal teardown.
os._exit(1 if fails else 0)
