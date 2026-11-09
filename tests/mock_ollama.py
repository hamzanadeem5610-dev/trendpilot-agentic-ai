"""A tiny fake Ollama server for offline tests (no GPU / no model needed).

It answers /api/version, /api/tags and /api/chat. For /api/chat it looks at the JSON schema the agent
sends in `format` to decide which tool is calling, and returns a plausible answer. Baseline (v1) prompts
get deliberately weak answers (too long, spam hashtags, 75-second scripts) so the reviewer loop and
the error analysis can be exercised end-to-end.

Run:  python tests/mock_ollama.py --port 11500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LONG = ("In today's fast-paced world, AI is a game-changer. " * 3 +
        "This project shows how we can unlock the power of technology to revolutionize everything. " * 9)


def topic_of(text: str) -> str:
    m = re.search(r"Topic: (.+)", text) or re.search(r"about:? (.+?)(?: in a| for |\.|$)", text)
    return (m.group(1) if m else "my AI project").strip()[:60]


def answer(payload: dict) -> str:
    fmt = payload.get("format")
    keys = set((fmt or {}).get("properties", {}).keys()) if isinstance(fmt, dict) else set()
    msgs = payload.get("messages", [])
    system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    user = msgs[-1]["content"] if msgs else ""
    v1 = system.startswith("You are a helpful assistant")
    improving = "IMPORTANT - improve" in user or "The user asked" in user
    topic = topic_of(user)
    words = [w for w in re.findall(r"[A-Za-z0-9]+", topic) if len(w) > 2][:3] or ["AI"]
    rnd = random.Random(hash(user) % 10_000)

    if not keys:                                    # plain text (connection test)
        return "1. My first week as an AI intern\n2. What I learned shipping a model\n3. Mistakes I made\n4. Tools I use daily\n5. Advice for new interns"
    if "target_audience" in keys and "key_message" in keys:
        return json.dumps({"target_audience": "CS students and junior ML engineers", "key_message": f"Building {topic} teaches real ML skills",
                           "goal": "get recruiter attention", "specific_details": [f"built {topic}", "trained on Google Colab", "tested on unseen data"],
                           "avoid": ["generic hype", "invented metrics"]})
    if "steps" in keys:
        out = "post" if "post" in user.lower() else "script"
        steps = [{"tool": "trend_idea_generator", "reason": "find the best angle"}]
        if "caption_writer" in user and out == "post":
            steps += [{"tool": "caption_writer", "reason": "write post"}, {"tool": "hashtag_generator", "reason": "tags"}]
        steps += [{"tool": "reel_script_generator", "reason": "video"}, {"tool": "made_up_tool", "reason": "x"},
                  {"tool": "content_reviewer", "reason": "check"}, {"tool": "file_saver", "reason": "save"}]
        return json.dumps({"steps": steps})
    if "angles" in keys:
        angles = [{"title": f"{t} angle on {topic}", "angle": t, "hook_idea": f"I built {topic} and learned one thing",
                   "why_it_works": "relatable", "score": s} for t, s in
                  [("Lesson learned", 9), ("Behind the scenes", 7), ("Mistake I made", 8), ("How-to", 6), ("Contrarian take", 7)]]
        return json.dumps({"angles": angles, "best_index": 0, "target_audience": "CS students and recruiters"})
    if "caption" in keys:
        if v1:
            return json.dumps({"hook": "Excited to share my new project with everyone today, it was a long journey!", "caption": LONG,
                               "call_to_action": "Like and share!"})
        good = improving or rnd.random() < 0.5
        cap = (f"I built {topic}. The hardest part was not the model - it was the data.\n\n"
               + "I collected, cleaned and labelled images, then trained on Google Colab and tested on unseen photos. " * (3 if good else 1)
               + ("\n\nThe biggest lesson: check your labels before you tune anything." if good else " It was a game-changer."))
        return json.dumps({"hook": f"I built {topic}. The hardest part was not the model.", "caption": cap,
                           "call_to_action": "What was the hardest part of your last project?"})
    if "hashtags" in keys:
        if v1:
            return json.dumps({"hashtags": ["#viral", "#love", "#instagood", "#AI", "#tech", "#trending", "#follow", "#explore",
                                            "#coding", "#ml", "#deeplearning", "#data", "#python", "#photooftheday"]})
        return json.dumps({"hashtags": ["#AI", "#MachineLearning", "#" + "".join(w.title() for w in words), "#ComputerVision", "#Python"]})
    if "scenes" in keys:
        n = 7 if v1 else 5
        step = 11 if v1 else 6
        scenes = [{"time": f"{i * step}-{(i + 1) * step}s", "visual": f"Shot {i + 1} of the project",
                   "voiceover": f"Scene {i + 1}: here is how {topic} works step by step.", "on_screen_text": f"Step {i + 1}"} for i in range(n)]
        return json.dumps({"total_seconds": n * step, "scenes": scenes, "closing_line": "Follow for part 2!"})
    if "titles" in keys:
        return json.dumps({"titles": [f"I Built {topic}", f"{topic} in 30 Seconds", f"What {topic} Taught Me"],
                           "best_title": f"I Built {topic}"[:58], "thumbnail_text": "AI in Action"})
    if "image_prompt" in keys:
        return json.dumps({"image_prompt": f"Clean flat illustration of {topic}, laptop with bounding boxes, blue palette, 4:5",
                           "alt_text": f"Illustration of {topic}"})
    if "verdict" in keys and "hook_strength" in json.dumps(fmt):
        bad = "game-changer" in user or "Automatic checks already found" in user
        sc = 5.5 if bad else 8.4
        return json.dumps({"scores": {k: sc for k in ("hook_strength", "clarity", "length_fit", "tone_fit", "hashtag_relevance", "platform_readiness")},
                           "overall": sc, "verdict": "needs_revision" if bad else "ready",
                           "issues": ["Caption uses cliches"] if bad else [], "suggestions": ["Add one concrete metric placeholder like [your mAP score]"],
                           "rewrite": ["caption_writer"] if bad else []})
    if "tools" in keys and "instruction" in keys:
        return json.dumps({"tools": ["caption_writer"], "instruction": "apply the user feedback"})
    if "days" in keys:
        d = int(re.search(r"Days available: (\d+)", user).group(1)) if "Days available" in user else 3
        return json.dumps({"title": f"Master {topic} in {d} days", "days": [
            {"day": i + 1, "focus": f"Part {i + 1}", "goals": ["understand basics"], "tasks": ["read docs", "code example", "practise"],
             "minutes": 90, "resources": [{"name": "Official documentation", "type": "docs"}]} for i in range(d)]})
    if "key_concepts" in keys:
        return json.dumps({"summary": f"{topic} summary.", "key_concepts": [{"term": f"Concept {i}", "explanation": "simple explanation"} for i in range(6)],
                           "cheat_sheet": ["fact 1", "fact 2"], "common_mistakes": ["mistake 1"]})
    if "questions" in keys:
        return json.dumps({"questions": [{"question": f"Question {i}?", "options": ["A", "B", "C", "D"], "answer_index": i % 4,
                                          "explanation": "because", "difficulty": "medium"} for i in range(8)]})
    if "coverage" in json.dumps(fmt):
        return json.dumps({"scores": {"coverage": 8, "pacing": 8, "clarity": 8, "quiz_quality": 8}, "overall": 8, "verdict": "ready",
                           "issues": [], "suggestions": ["Add a mini project"], "rewrite": []})
    return json.dumps({k: "x" for k in keys})


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "mock-0.0"})
        if self.path == "/api/tags":
            return self._send({"models": [{"name": "gemma3:4b"}]})
        return self._send({"error": "not found"}, 404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/chat":
            text = answer(payload)
            return self._send({"message": {"role": "assistant", "content": text}, "prompt_eval_count": 300,
                               "eval_count": len(text) // 4, "done": True})
        if self.path == "/api/generate":
            return self._send({"response": answer({"messages": [{"role": "user", "content": payload.get("prompt", "")}]}), "done": True})
        return self._send({"error": "not found"}, 404)


def start(port: int = 11500) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11500)
    args = ap.parse_args()
    print(f"Mock Ollama on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
