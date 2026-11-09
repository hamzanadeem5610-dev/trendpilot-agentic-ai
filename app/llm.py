"""Thin, robust client for a local Ollama server.

* Uses Ollama's REST API (`/api/chat`) with `stream=False`, exactly like the brief's `ask_ollama` example.
* `generate_json` asks Ollama for **structured output** (a JSON schema in `format`), then validates the
  reply; if the model still returns broken JSON it retries with the parse error fed back to the model.
* Every call is logged (tool name, latency, tokens, success) - these logs are the agent's evaluation data.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import Settings


class OllamaError(RuntimeError):
    pass


ESSENTIAL_LISTS = {"angles", "hashtags", "scenes", "titles", "days", "questions", "key_concepts", "steps"}


@dataclass
class CallLog:
    tool: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True
    attempts: int = 1
    error: str = ""


@dataclass
class OllamaClient:
    settings: Settings = field(default_factory=Settings)
    calls: list = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def model(self) -> str:
        return self.settings.model

    def _url(self, path: str) -> str:
        return self.settings.ollama_host.rstrip("/") + path

    def health(self) -> dict:
        """Return server version + installed models, or raise OllamaError with a helpful message."""
        try:
            version = requests.get(self._url("/api/version"), timeout=5).json().get("version", "?")
            tags = requests.get(self._url("/api/tags"), timeout=5).json().get("models", [])
        except requests.RequestException as e:
            raise OllamaError(f"Ollama is not reachable at {self.settings.ollama_host}. Start it with `ollama serve`. ({e})")
        names = [m.get("name", "") for m in tags]
        return {"version": version, "models": names, "model_ready": any(n.split(":latest")[0] == self.model or n == self.model for n in names)}

    def ensure_model(self) -> str:
        """Use the configured model, or fall back to an installed small model (brief: Llama 3.2 3B, Qwen 2.5 3B, Phi-3 Mini)."""
        info = self.health()
        if info["model_ready"]:
            return self.model
        for alt in self.settings.fallback_models:
            if alt in info["models"]:
                self.settings.model = alt
                return alt
        raise OllamaError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")

    # ------------------------------------------------------------------ raw call
    def chat(self, prompt: str, system: str | None = None, fmt: Any = None, temperature: float | None = None,
             max_tokens: int | None = None, tool: str = "llm") -> tuple[str, dict]:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": self.settings.temperature if temperature is None else temperature,
                        "num_ctx": self.settings.num_ctx},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if fmt is not None:
            payload["format"] = fmt
        t0 = time.time()
        try:
            r = requests.post(self._url("/api/chat"), json=payload, timeout=self.settings.request_timeout)
        except requests.RequestException as e:
            raise OllamaError(f"Request to Ollama failed: {e}")
        if r.status_code >= 400:
            raise OllamaError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = data.get("message", {}).get("content", "")
        meta = {"seconds": round(time.time() - t0, 2), "prompt_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0)}
        return text, meta

    def generate(self, prompt: str, system: str | None = None, tool: str = "llm", **kw) -> str:
        """Plain text generation (used for the brief's basic connection test)."""
        text, meta = self.chat(prompt, system=system, tool=tool, **kw)
        self.calls.append(CallLog(tool, meta["seconds"], meta["prompt_tokens"], meta["output_tokens"]))
        return text

    # ------------------------------------------------------------------ structured JSON
    @staticmethod
    def parse_json(text: str) -> dict:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start:end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)        # trailing commas
            return json.loads(candidate)
        raise json.JSONDecodeError("No JSON object found", text, 0)

    def generate_json(self, prompt: str, system: str, schema: dict, tool: str,
                      temperature: float | None = None, max_tokens: int | None = 900,
                      required: list | None = None) -> dict:
        """Ask for JSON that follows `schema`; validate required keys; retry with error feedback."""
        required = required or list(schema.get("required", []))
        fmt: Any = schema
        last_err = ""
        total = CallLog(tool, 0.0)
        for attempt in range(1, self.settings.json_retries + 2):
            p = prompt if not last_err else (
                f"{prompt}\n\nYour previous answer could not be used ({last_err}). "
                "Reply again with ONLY one valid JSON object that matches the schema.")
            try:
                text, meta = self.chat(p, system=system, fmt=fmt, temperature=temperature, max_tokens=max_tokens, tool=tool)
            except OllamaError as e:
                if fmt is not schema or "format" not in str(e).lower():
                    total.ok, total.error, total.attempts = False, str(e), attempt
                    self.calls.append(total)
                    raise
                fmt = "json"          # older Ollama: no JSON-schema support -> plain JSON mode
                last_err = "schema format unsupported"
                continue
            total.seconds += meta["seconds"]; total.prompt_tokens += meta["prompt_tokens"]; total.output_tokens += meta["output_tokens"]
            try:
                data = self.parse_json(text)
                if not isinstance(data, dict):
                    raise ValueError("reply is not a JSON object")
                # keys must exist; core content lists must not be empty (empty 'issues' lists are fine)
                missing = [k for k in required if k not in data or data[k] is None or data[k] == ""
                           or (k in ESSENTIAL_LISTS and not data[k])]
                if missing:
                    raise ValueError(f"missing or empty keys: {missing}")
                total.attempts = attempt
                self.calls.append(total)
                return data
            except (json.JSONDecodeError, ValueError) as e:
                last_err = str(e)[:160]
        total.ok, total.error, total.attempts = False, last_err, self.settings.json_retries + 1
        self.calls.append(total)
        raise OllamaError(f"[{tool}] model did not return valid JSON after {total.attempts} attempts: {last_err}")

    # ------------------------------------------------------------------ stats
    def stats(self, since: int = 0) -> dict:
        calls = self.calls[since:]
        return {"llm_calls": len(calls), "seconds": round(sum(c.seconds for c in calls), 1),
                "output_tokens": sum(c.output_tokens for c in calls),
                "retries": sum(c.attempts - 1 for c in calls), "failures": sum(not c.ok for c in calls)}
