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
os.environ["ASKSAGE_API_KEY"] = "ASKTOKEN"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import backend_switchboard._router as cr  # noqa: E402

SEEN = {}
CALLS = {}      # (mock name, upstream model) -> number of requests
REPLY = {}      # upstream model -> {"text": ..., "tool": {...}, "empty": True}; unscripted models get the plain stream


def scripted(spec, stream):
    """An Anthropic message (JSON or SSE) for a REPLY spec."""
    content = []
    if spec.get("text"):
        content.append({"type": "text", "text": spec["text"]})
    if spec.get("tool"):
        content.append({"type": "tool_use", "id": "tu_1", "name": spec["tool"]["name"], "input": spec["tool"]["input"]})
    msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": content,
           "stop_reason": "tool_use" if spec.get("tool") else "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
    if stream:
        import backend_switchboard._asksage as _a
        return _a.sse(msg), "text/event-stream"
    return json.dumps(msg).encode(), "application/json"


def mock(name):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.loads(raw)
            SEEN[name] = {"path": self.path, "h": {k.lower(): v for k, v in self.headers.items()}, "raw": raw, "body": body}
            CALLS[(name, body.get("model"))] = CALLS.get((name, body.get("model")), 0) + 1
            spec = REPLY.get("__merge") if b"<draft_1>" in raw else REPLY.get(body.get("model"))
            if spec is not None and not self.path.endswith("/query"):
                out, ctype = scripted(spec, body.get("stream"))
                self.send_response(200)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                return self.wfile.write(out)
            if body["model"] == "boom":
                out = b'{"type":"error","error":{"type":"api_error","message":"overloaded"}}'
                self.send_response(503)
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                return self.wfile.write(out)
            if self.path.endswith("/query"):
                out = json.dumps({"message": "\n\nhello", "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_time", "arguments": "{\"tz\": \"UTC\"}"}}] if body.get("tools") else None}).encode()
                self.send_response(200)
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


urls = {n: mock(n) for n in ("anthropic", "mag", "argo", "alcf", "asksage")}
argo_settings = os.path.join(tmp, "argo_settings.json")
json.dump({"apiKeyHelper": "echo 'ARGOTOKEN'", "env": {"ANTHROPIC_BASE_URL": urls["argo"] + "/argoapi"}},
          open(argo_settings, "w"))
D1 = {"features": {"expert": 0.0, "difficulty": 1.0}, "fail": False, "calls": 0, "texts": []}


class D1Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
        if self.path == "/judge":
            out = json.dumps({"features": D1.get("judge", {})}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            return self.wfile.write(out)
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
           "asksage_url": urls["asksage"],
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

asktools = [{"name": "get_time", "description": "t", "input_schema": {"type": "object"}}]
asked = {"system": [{"type": "text", "text": "SYS"}], "tools": asktools, "stream": True,
         "messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "get_time", "input": {"tz": "UTC"}}]},
                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "12:00"}, {"type": "text", "text": "thanks"}]}]}
st, out, _, _ = post("gpt-4.1@asksage", extra=asked)
a = SEEN["asksage"]
check("@asksage: /query with x-access-tokens, tools as a list, system_prompt, history turns",
      a["path"] == "/query" and a["h"].get("x-access-tokens") == "ASKTOKEN" and a["body"]["model"] == "gpt-4.1"
      and a["body"]["system_prompt"] == "SYS" and a["body"]["tools"][0]["function"]["name"] == "get_time"
      and [t["user"] for t in a["body"]["message"]] == ["me", "gpt", "me"]
      and "Result of that tool call:)\n12:00" in a["body"]["message"][2]["message"])
check("@asksage: tool call comes back as an Anthropic SSE tool_use stream",
      st == 200 and b"message_stop" in out and b'"stop_reason": "tool_use"' in out and b'"partial_json": "{\\"tz\\": \\"UTC\\"}"' in out, out[-300:])
st, out, _, _ = post("gpt-4.1@asksage", extra={"messages": [{"role": "user", "content": "hi"}]})
check("@asksage: plain answer as a non-streaming message", st == 200 and json.loads(out)["content"][0]["text"] == "hello" and json.loads(out)["stop_reason"] == "end_turn", out)

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

from backend_switchboard import _asksage as _ask  # noqa: E402
m, _ = _ask.to_message({"message": "Done. [tool_use functions.get_time id=x] {\"tz\": \"UTC\"} bye [tool_use nope] {}"},
                       {"model": "m", "tools": asktools}, {"message": ""})
check("@asksage: tool call written as text is recovered (known tools only)",
      [b["type"] for b in m["content"]] == ["text", "tool_use"] and m["content"][1]["input"] == {"tz": "UTC"}
      and m["stop_reason"] == "tool_use" and "[tool_use nope]" in m["content"][0]["text"], str(m))
post("claude-x@asksage", extra=ts)
check("tool search flattened for @asksage", [t["function"]["name"] for t in SEEN["asksage"]["body"]["tools"]] == ["Read"]
      and "Tool loaded: WebFetch" in SEEN["asksage"]["body"]["message"][-1]["message"])

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
n0 = len(log_lines(0))
s = ask("a fresh prompt while d1 is down")
check("auto: d1 down -> heuristic decides, request still served", s == 200 and SEEN["mag"]["body"]["model"] == "claude-small")
check("auto: heuristic source is logged", log_lines(n0 + 1)[-1]["decision"]["source"].startswith("heuristic"))
D1["fail"] = False
cr.Handler.auto.d1.until.clear()

cr.Handler.auto.c["tiers"] = {"small": ["boom@mag", "claude-small@argo"], "medium": TIERS["medium"], "large": TIERS["large"]}
n0 = len(log_lines(0))
s = ask("fallback prompt one")
check("auto: 503 from the first candidate -> next candidate answers", s == 200 and SEEN["argo"]["body"]["model"] == "claude-small")
last = log_lines(n0 + 1)[-1]
check("auto: failed attempt is recorded", last["attempts"] == [["boom@mag", 503]] and last["chosen"] == "claude-small@argo")
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


# --- verify + escalate ------------------------------------------------------------------------------------------------
LADDER = {"small": ["w-weak@mag"], "medium": ["w-mid@argo"], "large": ["w-strong@alcf"]}
A = cr.Handler.auto
A.c["tiers"] = LADDER
READ = {"name": "Read", "input_schema": {"type": "object", "required": ["file_path"]}}
GOOD = {"text": "Paris"}


def run(prompt, weak, mid=GOOD, strong=GOOD, tools=None, stream=False, prior=None, extra=None):
    REPLY.clear()
    REPLY.update({"w-weak": weak, "w-mid": mid, "w-strong": strong})
    CALLS.clear()
    D1["features"] = EASY
    msgs = list(prior or []) + [{"role": "user", "content": prompt}]
    body = {"messages": msgs, "stream": stream}
    if tools:
        body["tools"] = tools
    body.update(extra or {})
    s, out, _, _ = post("auto", extra=body)
    return s, out


def calls(name):
    return sum(n for (m, mod), n in CALLS.items() if mod == name)


A.c["max_escalations"] = 2
n_log = len(log_lines(0))
s, out = run("q-refusal", {"text": "I'm sorry, but I cannot help with that. I am unable to do this."})
check("verify: a refusal escalates and the stronger arm's answer is delivered",
      s == 200 and b"Paris" in out and calls("w-weak") == 1 and calls("w-mid") == 1 and calls("w-strong") == 0)
check("verify: the weak answer and its reason are logged", any(
    l.get("attempts") and l["attempts"][0][:3] == ["w-weak@mag", 200, ["refusal"]] and l.get("escalations") == 1
    for l in log_lines(n_log + 1)[n_log:]))
s, out = run("q-empty", {"text": ""})
check("verify: an empty answer escalates", s == 200 and b"Paris" in out and calls("w-mid") == 1)
s, out = run("q-unknown-tool", {"tool": {"name": "Nope", "input": {}}}, mid={"tool": {"name": "Read", "input": {"file_path": "a"}}}, tools=[READ])
check("verify: a call to a tool that was never offered escalates", s == 200 and b"Read" in out and calls("w-mid") == 1)
s, out = run("q-missing-arg", {"tool": {"name": "Read", "input": {}}}, mid={"tool": {"name": "Read", "input": {"file_path": "a"}}}, tools=[READ])
check("verify: a tool call missing a required argument escalates", s == 200 and calls("w-mid") == 1)
loop = {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "a"}}]}
res = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}
s, out = run("q-loop", {"tool": {"name": "Read", "input": {"file_path": "a"}}}, mid={"tool": {"name": "Read", "input": {"file_path": "b"}}},
             tools=[READ], prior=[loop, res, loop, res])
check("verify: repeating the same tool call a third time (a loop) escalates", s == 200 and calls("w-mid") == 1)
s, out = run("q-hedge", {"text": "I'm not sure about this. I might be wrong, but it could be Paris."})
check("verify: repeated hedging escalates", s == 200 and calls("w-mid") == 1)
s, out = run("q-fine", {"text": "Paris"})
check("verify: a good answer is delivered as is, nothing escalated", s == 200 and calls("w-weak") == 1 and calls("w-mid") == 0)
s, out = run("q-stream", {"text": "I cannot do that. I am unable to."}, stream=True)
check("verify: works on streamed answers and delivers the escalated stream", s == 200 and out.lstrip().startswith(b"event:") and b"Paris" in out)

s, out = run("q-pin", {"text": "I cannot do that. I am unable to."})
w0 = calls("w-weak")
REPLY.update({"w-weak": {"text": "I cannot do that. I am unable to."}, "w-mid": GOOD})
CALLS.clear()
post("auto", extra={"messages": [{"role": "user", "content": "q-pin"}]})
check("verify: after escalating, the rest of that user turn starts at the stronger arm (pinned)",
      w0 == 1 and calls("w-weak") == 0 and calls("w-mid") == 1)

A.c["max_escalations"] = 1
refuse = {"text": "I cannot do that. I am unable to."}
s, out = run("q-capped", refuse, mid=refuse)
check("verify: escalation is capped; the last answer is delivered rather than looping", s == 200 and calls("w-strong") == 0 and calls("w-mid") == 1)
A.c["max_escalations"] = 2
s, out = run("q-soft-cap", refuse, mid=refuse)
check("verify: if the stronger arm also says it cannot, stop (a real limit, not a weak model)",
      s == 200 and calls("w-weak") == 1 and calls("w-mid") == 1 and calls("w-strong") == 0)
s, out = run("q-hard-continues", {"text": ""}, mid={"text": ""})
check("verify: hard failures keep escalating past the first step", s == 200 and calls("w-strong") == 1 and b"Paris" in out)
A.c["verify"] = False
s, out = run("q-noverify", refuse)
check("verify: can be switched off", s == 200 and calls("w-mid") == 0 and b"cannot" in out)
A.c["verify"] = True

A.c["judge"] = {"enabled": True, "min_correct": 0.5, "max_refuses": 0.5}
D1["judge"] = {"correct": 0.2, "refuses": 0.0}
s, out = run("q-judge", {"text": "A confident but wrong answer."})
check("verify: a low d1 judge score escalates", s == 200 and calls("w-mid") == 1)
D1["judge"] = {"correct": 0.9, "refuses": 0.0}
s, out = run("q-judge-ok", {"text": "A confident right answer."})
check("verify: a high d1 judge score accepts", s == 200 and calls("w-mid") == 0)
A.c["judge"] = {"enabled": False}

part = auto_mod_cfg = cr.auto_mod.auto_config({"auto": {"synthesis": {"enabled": True}, "judge": {"enabled": True}}})
check("config: a partial nested option keeps the other defaults",
      part["synthesis"]["enabled"] and part["synthesis"]["merger"] == "muse" and part["judge"]["min_correct"] == 0.5)

# --- synthesis ---------------------------------------------------------------------------------------------------------
A.c["tiers"], A.c["enabled_arms"] = {}, ["muse", "beam", "sonnet"]
A.c["synthesis"] = dict(A.c["synthesis"], enabled=True, min_need=0.0, agree_above=0.6)
BEAM, MUSE = "beam-1.0-bf16", "muse-spark-bf16"


def synth(beam, muse, merged=None, prior=None, tools=None):
    REPLY.clear()
    REPLY.update({BEAM: beam, MUSE: muse})
    if merged:
        REPLY["__merge"] = merged
    CALLS.clear()
    D1["features"] = dict(EASY, **{"domain=chat": 1.0})
    body = {"messages": list(prior or []) + [{"role": "user", "content": "unique %s" % time.time()}]}
    if tools:
        body["tools"] = tools
    s, out, _, _ = post("auto", extra=body)
    return s, out


s, out = synth({"text": "The capital of France is Paris."}, {"text": "The capital of France is Paris."})
check("synthesis: two agreeing prose drafts -> one is delivered, no merge call", s == 200 and calls(BEAM) + calls(MUSE) == 2)
s, out = synth({"text": "The answer is alpha because of reason one."}, {"text": "Probably omega, owing to something else entirely."},
               merged={"text": "MERGED FINAL"})
check("synthesis: differing prose drafts are merged by the merger arm", s == 200 and b"MERGED FINAL" in out and calls(MUSE) == 2 and calls(BEAM) == 1)
mb = SEEN["alcf"]["body"]
check("synthesis: the merge request has no tools and carries both drafts",
      "tools" not in mb and "<draft_1>" in json.dumps(mb) and "alpha" in json.dumps(mb) and "omega" in json.dumps(mb))
s, out = synth({"tool": {"name": "Read", "input": {"file_path": "a"}}}, {"text": "I would just answer directly."}, tools=[READ])
check("synthesis: a tool call is never merged; the tool-calling draft is chosen", s == 200 and b"Read" in out and calls(BEAM) + calls(MUSE) == 2)
s, out = synth({"text": ""}, {"text": "Muse answers alone."})
check("synthesis: if one draft is unusable the other is delivered", s == 200 and b"Muse answers alone." in out)
tr = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "a"}}]},
      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}]
REPLY.clear()
REPLY.update({BEAM: GOOD, MUSE: GOOD})
CALLS.clear()
post("auto", extra={"messages": tr, "tools": [READ]})
check("synthesis: tool-result continuations are not fanned out", calls(BEAM) + calls(MUSE) == 1)
A.c["synthesis"] = dict(A.c["synthesis"], enabled=False)
A.c["enabled_arms"], A.c["tiers"] = None, TIERS
REPLY.clear()

print("\n%d failure(s)" % len(fails))
sys.stdout.flush()
sys.stderr.flush()
# Python 3.8 aborts at shutdown if a daemon handler thread is still logging to stderr, so skip normal teardown.
os._exit(1 if fails else 0)
