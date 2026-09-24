"""Optional OpenRouter backend for the frontier-model replication (Appendix I).

Set OPENROUTER_API_KEY in the environment. Reasoning models are mapped onto the think-model protocol:
    "<think>" + reasoning + "</think>\\n" + content        (finish_reason == "stop")
    "<think>" + reasoning                                  (finish_reason == "length": truncated -> no action)
so parsing and scoring are the ones used for the local models. Every call is appended to a JSONL log
(UTC time, requested/returned model, provider, payload, raw response, usage, cost) and cumulative spend
is tracked in a file so a batch aborts at API_HARD_CAP (default $90).
"""
import json, os, time, fcntl, urllib.request, urllib.error, datetime

HARD_CAP = float(os.environ.get("API_HARD_CAP", "90.0"))


class SpendCap(RuntimeError):
    pass


class OpenRouterGenerator:
    think_model = True

    def __init__(self, model, effort="high", max_tokens=16000, log_path=None, spend_path=None, provider=None):
        self.model, self.effort, self.max_tokens, self.provider = model, effort, max_tokens, provider
        self.log_path = log_path
        self.spend_path = spend_path or os.path.join(os.path.dirname(log_path) if log_path else ".", "spend.json")
        self.seed = None; self.tag = None
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("set OPENROUTER_API_KEY")

    def spent(self):
        try:
            return json.load(open(self.spend_path))["spent"]
        except Exception:
            return 0.0

    def _add_spend(self, cost):
        os.makedirs(os.path.dirname(os.path.abspath(self.spend_path)), exist_ok=True)
        with open(self.spend_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.seek(0)
            try:
                d = json.load(f)
            except Exception:
                d = {"spent": 0.0, "calls": 0}
            d["spent"] += float(cost); d["calls"] += 1
            f.seek(0); f.truncate(); json.dump(d, f); f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        return d["spent"]

    def chat(self, msgs, retries=6):
        if self.spent() >= HARD_CAP:
            raise SpendCap(f"hard cap ${HARD_CAP} reached (spent ${self.spent():.2f})")
        body = {"model": self.model, "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "max_tokens": self.max_tokens, "reasoning": {"effort": self.effort}, "usage": {"include": True}}
        if self.seed is not None:
            body["seed"] = int(self.seed)
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        data = json.dumps(body).encode()
        for att in range(retries):
            t0 = datetime.datetime.utcnow().isoformat() + "Z"
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data, headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
                r = json.load(urllib.request.urlopen(req, timeout=240))
                if "error" in r and not r.get("choices"):
                    raise urllib.error.HTTPError(None, 500, str(r["error"])[:200], None, None)
                ch = r["choices"][0]; msg = ch["message"]
                u = r.get("usage", {}) or {}; cost = float(u.get("cost", 0.0) or 0.0)
                total = self._add_spend(cost)
                if self.log_path:
                    rec = dict(time=t0, tag=self.tag, model_requested=self.model, model_returned=r.get("model"),
                               provider=r.get("provider"), finish_reason=ch.get("finish_reason"), usage=u, cost=cost,
                               cumulative_spent=total, payload=body, raw=r)
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                reasoning = msg.get("reasoning") or ""
                if not reasoning and msg.get("reasoning_details"):
                    reasoning = "".join(d.get("text", "") for d in msg["reasoning_details"] if isinstance(d, dict))
                return dict(reasoning=reasoning, content=msg.get("content") or "", finish_reason=ch.get("finish_reason"))
            except urllib.error.HTTPError as e:
                if (getattr(e, "code", 0) in (408, 429) or getattr(e, "code", 0) >= 500) and att < retries - 1:
                    time.sleep(min(60, 5 * (att + 1))); continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if att < retries - 1:
                    time.sleep(min(60, 5 * (att + 1))); continue
                raise
        raise RuntimeError("unreachable")

    def __call__(self, msgs):
        out = self.chat(msgs)
        if out["finish_reason"] == "length":
            return "<think>" + out["reasoning"]
        return "<think>" + out["reasoning"] + "</think>\n" + out["content"]
