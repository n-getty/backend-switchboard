"""@copilot: GitHub Copilot's API (https://api.githubcopilot.com), authenticated with your `gh` login.

Copilot lists the endpoints each model supports. Claude models serve the Anthropic Messages API natively and are
forwarded by the router like any other native backend. Models that only offer OpenAI's /responses API (gpt-*, grok-*,
mai-*) are translated here, in process (stdlib only). Answers are requested whole and streaming is emulated.
Models that only offer /chat/completions (gemini-*) are not supported.
"""
import http.client
import json
import os
import re
import shutil
import ssl
import subprocess
import threading
import time
from urllib.parse import urlsplit

from ._asksage import _text, _tokens, sse

HEADERS = {"user-agent": "GitHubCopilot/1.0.105", "copilot-integration-id": "copilot-developer-cli",
           "x-initiator": "user"}
_tok = {"lock": threading.Lock(), "token": None, "at": 0.0}
_models = {"lock": threading.Lock(), "map": None, "at": 0.0}


def token():
    """COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN if set, else `gh auth token` (cached 2 min)."""
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    with _tok["lock"]:
        if _tok["token"] and time.time() - _tok["at"] < 120:
            return _tok["token"]
        exe = shutil.which("gh") or os.path.join(os.path.expanduser("~"), ".local/bin/gh")
        try:
            out = subprocess.run([exe, "auth", "token"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        _tok["token"], _tok["at"] = out.stdout.strip() or None, time.time()
        return _tok["token"]


def upstream_name(model):
    """Claude Code says claude-opus-5-5; Copilot says claude-opus-5.5."""
    if model.startswith("claude-"):
        return re.sub(r"-(\d+)-(\d+)(?=-|$)", r"-\1.\2", model)
    return model


def _conn(cfg, timeout):
    u = urlsplit(cfg["copilot_url"])
    if u.scheme == "https":
        return u, http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=ssl.create_default_context())
    return u, http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def _get(cfg, path, tok):
    u, conn = _conn(cfg, 30)
    try:
        conn.request("GET", u.path.rstrip("/") + path, headers=dict(HEADERS, authorization="Bearer " + tok))
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


def endpoint(cfg, model, tok):
    """'messages', 'responses' or 'chat' for this model, from Copilot's model list (cached 10 min)."""
    with _models["lock"]:
        if _models["map"] is None or time.time() - _models["at"] > 600:
            try:
                status, raw = _get(cfg, "/models", tok)
                data = json.loads(raw)["data"] if status == 200 else []
                m = {}
                for d in data:
                    eps = d.get("supported_endpoints") or []
                    m[d["id"]] = ("messages" if "/v1/messages" in eps else "responses" if "/responses" in eps
                                  else "chat" if "/chat/completions" in eps else None)
                _models["map"], _models["at"] = m, time.time()
            except (OSError, ValueError, KeyError, TypeError):
                _models["map"] = _models["map"] or {}
        hit = _models["map"].get(model)
    return hit or ("messages" if model.startswith("claude-") else "responses")


def _image(b):
    s = b.get("source") or {}
    url = s.get("url") if s.get("type") == "url" else "data:%s;base64,%s" % (s.get("media_type"), s.get("data"))
    return {"type": "input_image", "image_url": url}


def to_responses(obj):
    """Anthropic request body -> OpenAI Responses payload."""
    system = obj.get("system")
    if isinstance(system, list):
        system = "\n\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    items = []
    for m in obj.get("messages", []):
        content = m.get("content")
        asst = m.get("role") == "assistant"
        tt = "output_text" if asst else "input_text"
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []

        def flush():
            if parts:
                items.append({"role": "assistant" if asst else "user", "content": list(parts)})
                del parts[:]
        for b in content or []:
            t = b.get("type")
            if t == "text" and b.get("text"):
                parts.append({"type": tt, "text": b["text"]})
            elif t == "image" and not asst:
                parts.append(_image(b))
            elif t == "tool_use":
                flush()
                items.append({"type": "function_call", "call_id": b["id"], "name": b["name"],
                              "arguments": json.dumps(b.get("input", {}))})
            elif t == "tool_result":
                flush()
                out = _text(b.get("content"))
                items.append({"type": "function_call_output", "call_id": b["tool_use_id"],
                              "output": ("Error: " + out) if b.get("is_error") else out})
        flush()
    r = {"model": obj["model"], "input": items, "stream": False}
    if system:
        r["instructions"] = system
    if obj.get("max_tokens"):
        r["max_output_tokens"] = obj["max_tokens"]
    tools = [{"type": "function", "name": t["name"], "description": t.get("description", ""),
              "parameters": t.get("input_schema", {"type": "object"})}
             for t in obj.get("tools") or [] if isinstance(t, dict) and t.get("name")]
    if tools:
        r["tools"] = tools
        tc = obj.get("tool_choice") or {}
        if tc.get("type") == "any":
            r["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            r["tool_choice"] = {"type": "function", "name": tc.get("name")}
        elif tc.get("type") == "none":
            r["tool_choice"] = "none"
    return r


def to_message(d, req):
    """Responses result -> (Anthropic message dict, None) or (None, (status, type, message))."""
    if isinstance(d.get("error"), dict):
        e = d["error"]
        return None, (502, "api_error", e.get("message") or e.get("code") or "upstream error")
    content = []
    for it in d.get("output") or []:
        if it.get("type") == "message":
            text = "".join(c.get("text", "") for c in it.get("content") or [] if c.get("type") == "output_text").strip()
            if text:
                content.append({"type": "text", "text": text})
        elif it.get("type") == "function_call":
            try:
                args = json.loads(it.get("arguments") or "{}")
            except ValueError:
                args = {}
            content.append({"type": "tool_use", "id": it.get("call_id") or it.get("id"), "name": it.get("name"),
                            "input": args if isinstance(args, dict) else {}})
    tool = any(b["type"] == "tool_use" for b in content)
    cut = d.get("status") == "incomplete"
    u = d.get("usage") or {}
    return {"id": "msg_" + (d.get("id") or "x").replace("resp_", "")[:24], "type": "message", "role": "assistant",
            "model": req["model"], "content": content or [{"type": "text", "text": ""}],
            "stop_reason": "tool_use" if tool else "max_tokens" if cut else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": u.get("input_tokens") or _tokens(req.get("messages", [])),
                      "output_tokens": u.get("output_tokens") or _tokens(content)}}, None


def handle(h, cfg, obj, tok, model):
    """Serve one /v1/messages (or count_tokens) request on handler `h` via /responses; return the status sent."""
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

    if endpoint(cfg, model, tok) == "chat":
        return fail(400, "invalid_request_error", "%s only offers /chat/completions on Copilot, which @copilot does not support" % model)
    req = to_responses(obj)
    if h.path.split("?")[0].endswith("/count_tokens"):
        return send(200, json.dumps({"input_tokens": _tokens(req["input"]) + _tokens(req.get("tools", []))}).encode())
    u, conn = _conn(cfg, 900)
    try:
        conn.request("POST", u.path.rstrip("/") + "/responses", json.dumps(req),
                     dict(HEADERS, authorization="Bearer " + tok, **{"content-type": "application/json"}))
        r = conn.getresponse()
        raw = r.read()
        d = json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return fail(502, "api_error", "Copilot request failed: %s" % e)
    finally:
        conn.close()
    if r.status != 200:
        e = d.get("error") if isinstance(d, dict) else None
        msg = e.get("message") if isinstance(e, dict) else raw[:300].decode("utf8", "replace")
        return fail(r.status, "authentication_error" if r.status == 401 else "api_error", str(msg))
    msg, err = to_message(d, obj)
    if err:
        return fail(*err)
    if obj.get("stream"):
        return send(200, sse(msg), "text/event-stream")
    return send(200, json.dumps(msg).encode())
