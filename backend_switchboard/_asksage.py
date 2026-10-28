"""@asksage: translate Anthropic Messages requests to AskSage's /query API, in process (stdlib only).

AskSage takes a message history of {"user": "me"|"gpt", "message": text} turns plus an OpenAI-style `tools` list, and
returns the whole answer at once (`message`, `tool_calls`). It has no slot for tool results, so tool calls and their
results travel through the history as text turns; the model still sees the real `tools` list.
"""
import http.client
import json
import os
import ssl
import time
import uuid
from urllib.parse import urlsplit


def _text(content):
    """Flatten a tool_result's content (string or blocks) to text."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "[%s omitted]" % b.get("type")) if isinstance(b, dict) else str(b)
                     for b in content or [])


def _render(blk):
    if isinstance(blk, str):
        return blk
    t = blk.get("type")
    if t == "text":
        return blk.get("text", "")
    if t == "tool_use":
        return "(Earlier in this conversation I used the %s tool with input %s.)" % (blk.get("name"), json.dumps(blk.get("input", {})))
    if t == "tool_result":
        err = " (error)" if blk.get("is_error") else ""
        return "(Result of that tool call%s:)\n%s" % (err, _text(blk.get("content")))
    if t in ("thinking", "redacted_thinking"):
        return ""
    return "[%s omitted]" % t


def to_query(obj):
    """Anthropic request body -> AskSage /query payload."""
    system = obj.get("system")
    if isinstance(system, list):
        system = "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    turns = []
    for m in obj.get("messages", []):
        content = m.get("content")
        text = "\n\n".join(x for x in (_render(b) for b in content) if x) if isinstance(content, list) else content or ""
        who = "gpt" if m.get("role") == "assistant" else "me"
        if turns and turns[-1]["user"] == who:
            turns[-1]["message"] += "\n\n" + text
        else:
            turns.append({"user": who, "message": text})
    q = {"message": turns or "", "model": obj["model"], "persona": "default", "dataset": "none", "live": 0,
         "limit_references": 0}
    if system:
        q["system_prompt"] = system
    if obj.get("temperature") is not None:
        q["temperature"] = obj["temperature"]
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                               "parameters": t.get("input_schema", {"type": "object"})}}
             for t in obj.get("tools") or [] if isinstance(t, dict) and t.get("name")]
    if tools:  # a JSON list; a JSON-encoded string is silently ignored by most models
        q["tools"] = tools
        choice = (obj.get("tool_choice") or {}).get("type")
        if choice in ("any", "tool"):
            q["tool_choice"] = "required"
        elif choice == "none":
            q["tool_choice"] = "none"
    return q


def _tokens(x):
    return max(1, len(x if isinstance(x, str) else json.dumps(x)) // 4)


def _mimicked_calls(text, names):
    """Models sometimes copy the history's "[tool_use NAME id=..] {json}" notation instead of making a real call.
    Recover those as tool_use blocks (known tool names only); return (remaining text, blocks)."""
    dec, blocks, out, i = json.JSONDecoder(), [], [], 0
    while True:
        j = text.find("[tool_use ", i)
        k = text.find("]", j) if j >= 0 else -1
        if k < 0:
            out.append(text[i:])
            break
        head = text[j + 10:k].split()
        name = (head[0] if head else "").split("functions.")[-1]
        if name not in names:
            out.append(text[i:k + 1])
            i = k + 1
            continue
        args, end = {}, k + 1
        try:
            rest = text[k + 1:]
            val, n = dec.raw_decode(rest.lstrip())
            if isinstance(val, dict):
                args, end = val, k + 1 + (len(rest) - len(rest.lstrip())) + n
        except ValueError:
            pass
        out.append(text[i:j])
        blocks.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:24], "name": name, "input": args})
        i = end
    return "".join(out).strip(), blocks


def to_message(d, req, q):
    """AskSage /query response -> (Anthropic message dict, None) or (None, (status, type, message))."""
    err = d.get("error")
    if isinstance(err, dict):
        code = err.get("code", "")
        status = {"model_not_permitted": 403}.get(code, err.get("http_status") or 502)
        return None, (status, "permission_error" if status == 403 else "api_error", d.get("message") or code)
    content = []
    text = (d.get("message") or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    for c in d.get("tool_calls_unified") or d.get("tool_calls") or []:
        # OpenAI-shaped ({function: {name, arguments}}, arguments a JSON string or dict) or Anthropic-shaped ({name, input})
        fn = c.get("function") or c
        args = fn.get("arguments", fn.get("input")) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if fn.get("name"):
            content.append({"type": "tool_use", "id": str(c.get("id") or "toolu_" + uuid.uuid4().hex[:24]),
                            "name": fn["name"], "input": args if isinstance(args, dict) else {}})
    names = {t["name"] for t in req.get("tools") or [] if isinstance(t, dict) and t.get("name")}
    if "[tool_use " in text and not any(b["type"] == "tool_use" for b in content):
        text, found = _mimicked_calls(text, names)
        content = ([{"type": "text", "text": text}] if text else []) + found
    tool = any(b["type"] == "tool_use" for b in content)
    return {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": req["model"],
            "content": content or [{"type": "text", "text": ""}], "stop_reason": "tool_use" if tool else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": _tokens(q["message"]) + _tokens(q.get("tools", [])) + _tokens(q.get("system_prompt", "")),
                      "output_tokens": _tokens(content)}}, None


def sse(msg):
    """Whole message -> Anthropic SSE event stream."""
    def ev(name, data):
        return ("event: %s\ndata: %s\n\n" % (name, json.dumps(dict(data, type=name)))).encode()
    out = [ev("message_start", {"message": dict(msg, content=[], stop_reason=None,
                                                usage={"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 1})})]
    for i, b in enumerate(msg["content"]):
        if b["type"] == "tool_use":
            out.append(ev("content_block_start", {"index": i, "content_block": dict(b, input={})}))
            out.append(ev("content_block_delta", {"index": i, "delta": {"type": "input_json_delta",
                                                                         "partial_json": json.dumps(b["input"])}}))
        else:
            out.append(ev("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}}))
            out.append(ev("content_block_delta", {"index": i, "delta": {"type": "text_delta", "text": b["text"]}}))
        out.append(ev("content_block_stop", {"index": i}))
    out.append(ev("message_delta", {"delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                                    "usage": {"output_tokens": msg["usage"]["output_tokens"]}}))
    out.append(ev("message_stop", {}))
    return b"".join(out)


def _query(cfg, q, key):
    u = urlsplit(cfg["asksage_url"])
    if u.scheme == "https":
        ctx = ssl.create_default_context()
        cert = cfg.get("asksage_cert")
        if cert:
            ctx.load_verify_locations(os.path.expanduser(cert))
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=600, context=ctx)
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    try:
        conn.request("POST", u.path.rstrip("/") + "/query", json.dumps(q),
                     {"x-access-tokens": key, "content-type": "application/json"})
        r = conn.getresponse()
        raw = r.read()
        return r.status, raw
    finally:
        conn.close()


def handle(h, cfg, obj):
    """Serve one /v1/messages (or count_tokens) request on handler `h`; return the HTTP status sent."""
    def send(status, body, ctype="application/json"):
        h.send_response(status)
        h.send_header("content-type", ctype)
        h.send_header("content-length", str(len(body)))
        h.send_header("connection", "close")
        h.end_headers()
        h.wfile.write(body)
        h.close_connection = True
        return status

    def fail(status, etype, message):
        return send(status, json.dumps({"type": "error", "error": {"type": etype, "message": message}}).encode())

    q = to_query(obj)
    if h.path.split("?")[0].endswith("/count_tokens"):
        return send(200, json.dumps({"input_tokens": _tokens(q["message"]) + _tokens(q.get("tools", []))}).encode())
    key = os.environ.get("ASKSAGE_API_KEY")
    if not key:
        return fail(401, "authentication_error", "ASKSAGE_API_KEY is not set in the router's environment (restart it after exporting)")
    try:
        status, raw = _query(cfg, q, key)
        d = json.loads(raw)
    except (OSError, ValueError) as e:
        return fail(502, "api_error", "AskSage request failed: %s" % e)
    if status != 200:
        return fail(status, "api_error", str(d.get("message") if isinstance(d, dict) else raw[:300]))
    msg, err = to_message(d, obj, q)
    if err:
        return fail(*err)
    if obj.get("stream"):
        return send(200, sse(msg), "text/event-stream")
    return send(200, json.dumps(msg).encode())
