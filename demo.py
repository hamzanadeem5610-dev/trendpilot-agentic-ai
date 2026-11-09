"""TrendPilot - CLI demo (brief section 15, Option A).

Interactive:
    python demo.py
Non-interactive (used by the notebook / scripts):
    python demo.py --topic "YOLOv8 helmet detection project" --platform LinkedIn --tone "Professional but exciting" --output "Post + Reel Script"
StudyBuddy mode:
    python demo.py --mode study --topic "object detection" --days 7 --level Beginner --hours 2
Update study progress:
    python demo.py --progress 1 --day 2 --score 80
"""
from __future__ import annotations

import argparse
import sys

from app.agent import TrendPilotAgent
from app.config import OUTPUT_TYPES, PLATFORMS, TONES, Settings
from app.llm import OllamaError

BANNER = r"""
  _____                    _ ____  _ _       _
 |_   _| __ ___ _ __   __| |  _ \(_) | ___ | |_
   | || '__/ _ \ '_ \ / _` | |_) | | |/ _ \| __|
   | || | |  __/ | | | (_| |  __/| | | (_) | |_
   |_||_|  \___|_| |_|\__,_|_|   |_|_|\___/ \__|   Local Agentic AI (Ollama)
"""


def choose(label: str, options: list[str], default: str) -> str:
    print(f"\n{label}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    raw = input(f"Choose 1-{len(options)} or type your own [{default}]: ").strip()
    if not raw:
        return default
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw


def main() -> int:
    for stream in (sys.stdout, sys.stderr):          # Windows consoles: never crash on emoji / special characters
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="TrendPilot CLI demo")
    ap.add_argument("--mode", choices=["content", "study"], default="content")
    ap.add_argument("--topic"); ap.add_argument("--platform"); ap.add_argument("--tone"); ap.add_argument("--output")
    ap.add_argument("--days", type=int, default=7); ap.add_argument("--level", default="Beginner"); ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--model", help="Ollama model (default gemma3:4b)")
    ap.add_argument("--feedback", help="Optional improvement request applied after the first result")
    ap.add_argument("--progress", type=int, help="StudyBuddy plan id to update"); ap.add_argument("--day", type=int); ap.add_argument("--score", type=float)
    ap.add_argument("--history", action="store_true", help="Show memory (previous runs)")
    args = ap.parse_args()

    settings = Settings()
    if args.model:
        settings.model = args.model
    agent = TrendPilotAgent(settings)

    if args.history:
        for r in agent.memory.history():
            print(f"#{r['id']} {r['created_at']} | {r['topic']} | {r['platform']} | {r['tone']} | score {r['review_score']}")
        return 0
    if args.progress:
        p = agent.memory.update_progress(args.progress, args.day or 1, True, args.score)
        print(f"Plan {p['plan_id']} ({p['topic']}): {p['days_completed']}/{p['days_total']} days done ({p['percent']}%), avg quiz {p['avg_quiz_score']}")
        return 0

    print(BANNER)
    try:
        model = agent.llm.ensure_model()
        print(f"Ollama OK - model: {model}")
    except OllamaError as e:
        print(f"ERROR: {e}")
        return 1

    interactive = not args.topic
    if args.mode == "study":
        topic = args.topic or input("What do you want to learn? ").strip()
        days = args.days if not interactive else int(input("How many days? [7]: ") or 7)
        level = args.level if not interactive else (input("Your level (Beginner/Intermediate/Advanced) [Beginner]: ") or "Beginner")
        hours = args.hours if not interactive else float(input("Hours per day? [2]: ") or 2)
        res = agent.run_study(topic, days, level, hours)
    else:
        topic = args.topic or input("Enter your topic: ").strip()
        platform = args.platform or choose("Choose platform:", PLATFORMS, "LinkedIn")
        tone = args.tone or choose("Choose tone:", TONES, "Professional but exciting")
        output = args.output or choose("Choose output type:", OUTPUT_TYPES, "Post + hashtags")
        res = agent.run(topic, platform, tone, output)

    print("\n" + "=" * 70 + "\nFINAL OUTPUT\n" + "=" * 70)
    print(TrendPilotAgent.format_text(res))

    feedback = args.feedback
    while True:
        if feedback is None and interactive:
            feedback = input("\nWant improvements? Describe them (or press Enter to finish): ").strip()
        if not feedback:
            break
        res = agent.revise(res, feedback)
        print("\n" + "=" * 70 + "\nIMPROVED OUTPUT\n" + "=" * 70)
        print(TrendPilotAgent.format_text(res))
        feedback = None if interactive else ""
    print(f"\nSaved: {res['saved_files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
