"""Check a model's answer before the user sees it, so `auto` can escalate instead of handing back a weak result.

Cheap, deterministic checks run on every buffered answer:

  hard (always escalate)   empty answer; tool call for a tool the request never offered; tool arguments that are not an
                           object or miss a required key; the same tool call repeated again and again (a loop)
  soft (escalate)          the model says it cannot do the task, or hedges repeatedly ("I'm not sure", "I may be wrong")
  judge (optional)         a d1 judge's P(correct) / P(refuses) from the d1 service's /judge endpoint

Anything unparseable is accepted: the verifier only ever acts on evidence, and a wrongly escalated request costs money
while a wrongly accepted one is no worse than not routing at all.
"""
import json
import re

LOOP_REPEATS = 2           # a tool call already issued this many times in the history is a loop
HEDGE_LIMIT = 2            # hedging phrases in one answer before it counts as unconfident

_REFUSAL = re.compile(
    r"\b(i (?:can(?:'|’)?t|cannot|am unable to|(?:'|’)m unable to|am not able to|(?:'|’)m not able to)|"
    r"(?:this|that|it) is (?:beyond|outside) (?:my|what i)|beyond my (?:capabilit|ability)|"
    r"i (?:do not|don(?:'|’)t) have (?:the )?(?:ability|capability|access|enough (?:information|context)))", re.I)
_HEDGE = re.compile(
    r"\b(i(?:'|’)?m not (?:sure|certain|confident)|i am not (?:sure|certain|confident)|i (?:might|may) be wrong|"
    r"i (?:may|might) be mistaken|i(?:'|’)?m not 100%|i could be wrong|i (?:do not|don(?:'|’)t) know|"
    r"i(?:'|’)?m unsure|not entirely sure|i (?:can(?:'|’)?t|cannot) be (?:sure|certain))", re.I)


def parse(body, content_type=""):
    """Anthropic message (JSON or SSE) -> {"text", "tool_uses": [{"name", "input", "input_ok"}], "stop_reason"} or None."""
    if not body:
        return None
    if "event-stream" in content_type or body.lstrip().startswith(b"event:") or body.lstrip().startswith(b"data:"):
        return _parse_sse(body)
    try:
        d = json.loads(body)
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("type") == "error" or not isinstance(d.get("content"), list):
        return None
    text, tools = [], []
    for b in d["content"]:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            tools.append({"name": b.get("name"), "input": b.get("input"), "input_ok": isinstance(b.get("input"), dict)})
    return {"text": "".join(text), "tool_uses": tools, "stop_reason": d.get("stop_reason")}


def _parse_sse(body):
    blocks, stop, seen = {}, None, False
    for line in body.decode("utf8", "replace").split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        seen = seen or t == "message_start"
        if t == "content_block_start":
            blocks[ev.get("index", 0)] = {"b": dict(ev.get("content_block") or {}), "json": []}
        elif t == "content_block_delta" and ev.get("index", 0) in blocks:
            blk, dl = blocks[ev.get("index", 0)], ev.get("delta") or {}
            if dl.get("type") == "text_delta":
                blk["b"]["text"] = blk["b"].get("text", "") + dl.get("text", "")
            elif dl.get("type") == "input_json_delta":
                blk["json"].append(dl.get("partial_json", ""))
        elif t == "message_delta":
            stop = (ev.get("delta") or {}).get("stop_reason") or stop
        elif t == "error":
            return None
    if not seen:
        return None
    text, tools = [], []
    for i in sorted(blocks):
        b = blocks[i]["b"]
        if b.get("type") == "text":
            text.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            raw = "".join(blocks[i]["json"])
            try:
                inp, ok = (json.loads(raw) if raw else b.get("input") or {}), True
            except ValueError:
                inp, ok = None, False
            tools.append({"name": b.get("name"), "input": inp, "input_ok": ok and isinstance(inp, dict)})
    return {"text": "".join(text), "tool_uses": tools, "stop_reason": stop}


def _history_calls(req):
    """(name, canonical input) of every tool call the assistant already made in this conversation."""
    calls = []
    for m in req.get("messages", []):
        c = m.get("content") if isinstance(m, dict) and m.get("role") == "assistant" else None
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls.append((b.get("name"), json.dumps(b.get("input"), sort_keys=True)))
    return calls


def check(msg, req):
    """-> list of (code, severity, detail); empty means nothing wrong. severity is "hard" or "soft"."""
    if msg is None:
        return []
    out = []
    tools, text = msg["tool_uses"], msg["text"].strip()
    if not text and not tools:
        out.append(("empty", "hard", "no text and no tool call"))
    offered = {t.get("name"): t for t in req.get("tools") or [] if isinstance(t, dict)}
    history = _history_calls(req)
    for t in tools:
        if not t["input_ok"]:
            out.append(("bad_tool_input", "hard", "%s: arguments are not a JSON object" % t["name"]))
            continue
        spec = offered.get(t["name"])
        if offered and spec is None:
            out.append(("unknown_tool", "hard", "%s is not one of the offered tools" % t["name"]))
            continue
        required = ((spec or {}).get("input_schema") or {}).get("required") or []
        missing = [k for k in required if k not in t["input"]]
        if missing:
            out.append(("missing_required", "hard", "%s lacks %s" % (t["name"], ",".join(missing))))
        key = (t["name"], json.dumps(t["input"], sort_keys=True))
        if history.count(key) >= LOOP_REPEATS:
            out.append(("loop", "hard", "%s repeated the same call %d times" % (t["name"], history.count(key) + 1)))
    if text and not tools:
        if _REFUSAL.search(text):
            out.append(("refusal", "soft", "says it cannot do this"))
        if len(_HEDGE.findall(text)) >= HEDGE_LIMIT:
            out.append(("hedging", "soft", "repeatedly unsure"))
    return out


def similarity(a, b):
    """Rough agreement between two answers: Jaccard overlap of their word sets, 0..1."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0
