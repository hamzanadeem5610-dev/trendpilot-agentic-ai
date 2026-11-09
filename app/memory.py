"""Long-term memory for the agents (SQLite).

What is remembered
* every TrendPilot run: topic, platform, tone, hook, review score, saved file paths
* user preferences learned from usage (favourite platform / tone)
* StudyBuddy plans and day-by-day progress (completed days, quiz scores)

How it is used
* Before planning, the agent recalls the most similar previous topics (keyword overlap) and tells the
  writer tools what was already posted, so it does not repeat the same hook.
* The memory is exported to `outputs/memory_export.json` so it can be inspected without SQL.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "my", "your", "with", "about", "is", "how",
        "i", "it", "this", "that", "from", "using", "by", "at", "be", "as", "want", "create", "post", "viral"}


def _kw(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP and len(w) > 2}


class Memory:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, mode TEXT, topic TEXT, platform TEXT,
                tone TEXT, output_type TEXT, prompt_version TEXT, hook TEXT, review_score REAL, verdict TEXT,
                saved_md TEXT, saved_json TEXT, summary TEXT);
            CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, topic TEXT, days INTEGER, level TEXT,
                hours_per_day REAL, title TEXT, saved_md TEXT);
            CREATE TABLE IF NOT EXISTS study_progress (
                plan_id INTEGER, day INTEGER, completed INTEGER DEFAULT 0, quiz_score REAL, note TEXT,
                updated_at TEXT, PRIMARY KEY (plan_id, day));
            """)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _now():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ TrendPilot runs
    def save_run(self, result: dict) -> int:
        req, art, rev = result["request"], result.get("artifacts", {}), result.get("review", {})
        summary = {"content_idea": art.get("content_idea", {}).get("title") if isinstance(art.get("content_idea"), dict) else art.get("content_idea"),
                   "hashtags": art.get("hashtags", [])[:6]}
        with closing(self._conn()) as c, c:
            cur = c.execute(
                "INSERT INTO runs (created_at, mode, topic, platform, tone, output_type, prompt_version, hook, "
                "review_score, verdict, saved_md, saved_json, summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self._now(), "trendpilot", req["topic"], req["platform"], req["tone"], req["output_type"],
                 result.get("prompt_version", ""), art.get("hook", ""), rev.get("overall"), rev.get("verdict"),
                 result.get("saved_files", {}).get("markdown"), result.get("saved_files", {}).get("json"),
                 json.dumps(summary, ensure_ascii=False)))
            run_id = cur.lastrowid
        self._learn_preferences()
        return run_id

    def recall(self, topic: str, platform: str | None = None, k: int = 3) -> list[dict]:
        """Most similar previous runs (keyword Jaccard + small bonus for the same platform)."""
        q = _kw(topic)
        with closing(self._conn()) as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 200")]
        scored = []
        for r in rows:
            kw = _kw(r["topic"])
            sim = len(q & kw) / max(1, len(q | kw))
            if platform and r["platform"] == platform:
                sim += 0.1
            if sim > 0.15:
                scored.append((sim, r))
        return [dict(r, similarity=round(s, 2)) for s, r in sorted(scored, key=lambda x: -x[0])[:k]]

    def memory_notes(self, topic: str, platform: str | None = None) -> str:
        """Short text the writer tools receive, e.g. to avoid repeating a hook."""
        hits = self.recall(topic, platform)
        if not hits:
            return ""
        return " | ".join(f"Previous topic: {h['topic']} ({h['platform']}, {h['tone']}); hook used: \"{(h['hook'] or '')[:90]}\""
                          for h in hits) + " -> use a fresh angle and a different hook."

    def history(self, limit: int = 20) -> list[dict]:
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(
                "SELECT id, created_at, topic, platform, tone, output_type, review_score, verdict, saved_md "
                "FROM runs ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ preferences
    def set_pref(self, key: str, value: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR REPLACE INTO preferences VALUES (?,?,?)", (key, value, self._now()))

    def prefs(self) -> dict:
        with closing(self._conn()) as c:
            return {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM preferences")}

    def _learn_preferences(self) -> None:
        with closing(self._conn()) as c:
            rows = list(c.execute("SELECT platform, tone FROM runs ORDER BY id DESC LIMIT 30"))
        if rows:
            self.set_pref("favourite_platform", Counter(r["platform"] for r in rows).most_common(1)[0][0])
            self.set_pref("favourite_tone", Counter(r["tone"] for r in rows).most_common(1)[0][0])

    # ------------------------------------------------------------------ StudyBuddy
    def save_study_plan(self, req: dict, title: str, days: list, saved_md: str) -> int:
        with closing(self._conn()) as c, c:
            cur = c.execute("INSERT INTO study_plans (created_at, topic, days, level, hours_per_day, title, saved_md) "
                            "VALUES (?,?,?,?,?,?,?)", (self._now(), req["topic"], int(req["days"]), req["level"],
                                                       float(req["hours_per_day"]), title, saved_md))
            pid = cur.lastrowid
            for d in days:
                c.execute("INSERT OR IGNORE INTO study_progress (plan_id, day, completed, updated_at) VALUES (?,?,0,?)",
                          (pid, int(float(d.get("day", 0))), self._now()))
        return pid

    def set_plan_file(self, plan_id: int, saved_md: str) -> None:
        with closing(self._conn()) as c, c:
            c.execute("UPDATE study_plans SET saved_md=? WHERE id=?", (saved_md, plan_id))

    def update_progress(self, plan_id: int, day: int, completed: bool = True, quiz_score: float | None = None,
                        note: str = "") -> dict:
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR REPLACE INTO study_progress VALUES (?,?,?,?,?,?)",
                      (plan_id, day, int(completed), quiz_score, note, self._now()))
        return self.progress(plan_id)

    def progress(self, plan_id: int) -> dict:
        with closing(self._conn()) as c:
            plan = c.execute("SELECT * FROM study_plans WHERE id=?", (plan_id,)).fetchone()
            rows = [dict(r) for r in c.execute("SELECT * FROM study_progress WHERE plan_id=? ORDER BY day", (plan_id,))]
        done = sum(r["completed"] for r in rows)
        scores = [r["quiz_score"] for r in rows if r["quiz_score"] is not None]
        return {"plan_id": plan_id, "topic": plan["topic"] if plan else "?", "days_total": len(rows),
                "days_completed": done, "percent": round(100 * done / max(1, len(rows))),
                "avg_quiz_score": round(sum(scores) / len(scores), 1) if scores else None, "days": rows}

    def study_plans(self) -> list[dict]:
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute("SELECT * FROM study_plans ORDER BY id DESC")]

    def recall_study(self, topic: str) -> str:
        q = _kw(topic)
        notes = []
        for p in self.study_plans()[:50]:
            if len(q & _kw(p["topic"])) / max(1, len(q | _kw(p["topic"]))) > 0.2:
                pr = self.progress(p["id"])
                notes.append(f"Earlier plan '{p['title']}' ({p['days']} days): {pr['days_completed']}/{pr['days_total']} days done"
                             + (f", avg quiz {pr['avg_quiz_score']}" if pr["avg_quiz_score"] is not None else ""))
        return " | ".join(notes[:2])

    # ------------------------------------------------------------------ export
    def export_json(self, path: str | Path) -> Path:
        with closing(self._conn()) as c:
            data = {t: [dict(r) for r in c.execute(f"SELECT * FROM {t}")]
                    for t in ("runs", "preferences", "study_plans", "study_progress")}
        p = Path(path)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p
