"""Deterministic quality checks ("validation checks" from the brief's improvement list).

LLM reviewers are useful but can be lenient, so the agent also runs hard rules: length limits per
platform, hashtag counts and spam tags, reel duration, title length, cliches and possibly invented metrics.
Each issue carries the tool that should fix it - the reviewer loop uses this to decide what to re-run.
"""
from __future__ import annotations

import re

from .prompts import GENERIC_PHRASES, PLATFORM_RULES, platform_key

SPAM_TAGS = {"#love", "#instagood", "#photooftheday", "#follow", "#like4like", "#followforfollow", "#viral",
             "#trending", "#explore", "#instadaily", "#likeforlikes", "#picoftheday"}
STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "my", "your", "with", "about", "project",
        "is", "how", "i", "it", "this", "that", "from", "using", "by", "at", "be", "as"}


def parts_for(output_type: str, platform: str) -> list[str]:
    """Which content parts a request needs (drives planning and validation)."""
    o = (output_type or "").lower()
    p = platform_key(platform)
    if "full" in o or "plan" in o or "everything" in o:
        return ["caption", "hashtags", "reel_script", "titles"]
    parts = []
    if any(k in o for k in ("post", "caption", "short")):
        parts += ["caption", "hashtags"]
    if any(k in o for k in ("reel", "script", "video")):
        parts.append("reel_script")
        if p in ("YouTube Shorts", "TikTok", "Instagram Reels") or "video" in o:
            parts.append("titles")
    if not parts:
        parts = ["caption", "hashtags"]
    if "reel_script" in parts and "caption" not in parts:
        parts.append("hashtags")      # short videos still need hashtags in their description
    return list(dict.fromkeys(parts))


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'#@-]+\b", text or ""))


REQUEST_PREFIX = re.compile(
    r"^\s*(i\s+(want|need|would like)\s+to\s+|please\s+|can you\s+|help me\s+)?"
    r"(create|write|make|generate|draft|build|plan)\b.*?\b(about|on|for)\s+(my|our|the|a|an)?\s*", re.I)


def core_topic(text: str) -> str:
    """'I want to create a viral LinkedIn post about my YOLOv8 helmet detection project' -> 'YOLOv8 helmet detection project'."""
    short = REQUEST_PREFIX.sub("", text.strip(), count=1).strip(" .!?")
    return short if len(short) >= 3 else text.strip()


def topic_keywords(topic: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", topic.lower()) if w not in STOP and len(w) > 2}


def check_content(req: dict, art: dict) -> list[dict]:
    issues = []
    add = lambda code, tool, msg, severity="major": issues.append({"code": code, "tool": tool, "message": msg, "severity": severity})
    p = platform_key(req["platform"])
    rules = PLATFORM_RULES[p]
    needed = parts_for(req["output_type"], req["platform"])

    # ---- caption ----
    if "caption" in needed:
        cap = (art.get("caption") or "").strip()
        hook = (art.get("hook") or "").strip()
        if not cap:
            add("missing_caption", "caption_writer", "No caption was produced.")
        else:
            n = words(cap)
            lo, hi = rules["caption_words"]
            if n > hi * 1.15:
                add("too_long", "caption_writer", f"Caption has {n} words; {p} target is {lo}-{hi}.")
            elif n < lo * 0.6:
                add("too_short", "caption_writer", f"Caption has only {n} words; {p} target is {lo}-{hi}.", "minor")
            total_chars = len(cap) + sum(len(h) + 1 for h in art.get("hashtags", []))
            if total_chars > rules["max_chars"]:
                add("over_char_limit", "caption_writer", f"{total_chars} characters incl. hashtags > {p} limit {rules['max_chars']}.")
            low = cap.lower()
            found = [g for g in GENERIC_PHRASES if g in low]
            if found:
                add("generic_phrases", "caption_writer", f"Generic cliches used: {found}.")
            if re.search(r"#\w", cap) and p != "X / Twitter":
                add("hashtags_in_caption", "caption_writer", "Hashtags are mixed into the caption text.", "minor")
            nums = re.findall(r"\b\d+(?:\.\d+)?%|\b0\.\d+\b|\b\d{2,}x\b", cap)
            invented = [x for x in nums if x not in req["topic"]]
            if invented:
                add("possible_invented_metric", "caption_writer", f"Numbers not given by the user: {invented[:3]} - may be invented.", "minor")
        if not hook:
            add("missing_hook", "caption_writer", "No hook was produced.")
        elif words(hook) > 25:
            add("weak_hook", "caption_writer", f"Hook is {words(hook)} words - too long to grab attention.")
        elif hook.lower().startswith(("excited to share", "i am excited", "i'm excited", "hello everyone", "hi everyone")):
            add("weak_hook", "caption_writer", "Hook starts with a generic opener.")

    # ---- hashtags ----
    if "hashtags" in needed:
        tags = art.get("hashtags") or []
        lo, hi = rules["hashtags"]
        if not tags:
            add("missing_hashtags", "hashtag_generator", "No hashtags were produced.")
        else:
            if len(tags) > hi:
                add("too_many_hashtags", "hashtag_generator", f"{len(tags)} hashtags; {p} works best with {lo}-{hi}.")
            elif len(tags) < lo:
                add("too_few_hashtags", "hashtag_generator", f"Only {len(tags)} hashtags; {p} works best with {lo}-{hi}.", "minor")
            spam = [t for t in tags if t.lower() in SPAM_TAGS]
            if spam:
                add("irrelevant_hashtags", "hashtag_generator", f"Spam/generic hashtags: {spam}.")
            kw = topic_keywords(req.get("core_topic") or req["topic"])
            joined = " ".join(tags).lower()
            if kw and not any(k in joined for k in kw):
                add("irrelevant_hashtags", "hashtag_generator", "No hashtag mentions the actual topic.")

    # ---- reel script ----
    if "reel_script" in needed:
        sc = art.get("reel_script") or {}
        scenes = sc.get("scenes") or []
        lo, hi = rules["script_seconds"]
        if not scenes:
            add("missing_script", "reel_script_generator", "No reel script scenes were produced.")
        else:
            secs = script_seconds(sc)
            if secs > hi + 2:
                add("script_too_long", "reel_script_generator", f"Script runs ~{secs:.0f}s; limit for {p} is {hi}s.")
            elif secs < lo - 2:
                add("script_too_short", "reel_script_generator", f"Script runs ~{secs:.0f}s; minimum is {lo}s.", "minor")
            if len(scenes) < 3:
                add("script_too_few_scenes", "reel_script_generator", "Script needs an opening hook, explanation and closing (>= 3 scenes).")
            long_vo = [i + 1 for i, s in enumerate(scenes) if words(s.get("voiceover", "")) > 35]
            if long_vo:
                add("voiceover_too_long", "reel_script_generator", f"Voiceover too long in scene(s) {long_vo}.", "minor")
            if not (sc.get("closing_line") or "").strip():
                add("missing_closing", "reel_script_generator", "No closing line.", "minor")

    # ---- titles ----
    if "titles" in needed:
        t = art.get("titles") or {}
        if not t.get("best_title"):
            add("missing_title", "title_thumbnail_generator", "No title was produced.")
        elif len(t["best_title"]) > 70:
            add("title_too_long", "title_thumbnail_generator", f"Title is {len(t['best_title'])} characters (max 60).")
        if t.get("thumbnail_text") and words(t["thumbnail_text"]) > 6:
            add("thumbnail_too_long", "title_thumbnail_generator", "Thumbnail text should be at most 5 words.")
    return issues


def script_seconds(script: dict) -> float:
    """Duration of a script: max of total_seconds and the end time parsed from the last scene."""
    end = 0.0
    for s in script.get("scenes") or []:
        m = re.findall(r"(\d+(?:\.\d+)?)", str(s.get("time", "")))
        if m:
            end = max(end, float(m[-1]))
    try:
        declared = float(script.get("total_seconds") or 0)
    except (TypeError, ValueError):
        declared = 0.0
    return max(end, declared)


def check_study(req: dict, art: dict) -> list[dict]:
    issues = []
    add = lambda code, tool, msg, severity="major": issues.append({"code": code, "tool": tool, "message": msg, "severity": severity})
    plan = art.get("study_plan") or {}
    days = plan.get("days") or []
    if len(days) != int(req["days"]):
        add("wrong_day_count", "study_planner", f"Plan has {len(days)} days, user asked for {req['days']}.")
    budget = float(req["hours_per_day"]) * 60 * 1.2
    heavy = [d.get("day") for d in days if float(d.get("minutes") or 0) > budget]
    if heavy:
        add("overloaded_days", "study_planner", f"Days {heavy} exceed the daily time budget.")
    if any(not d.get("tasks") for d in days):
        add("empty_tasks", "study_planner", "Some days have no tasks.")
    quiz = (art.get("quiz") or {}).get("questions") or []
    if len(quiz) < 5:
        add("too_few_questions", "quiz_generator", f"Only {len(quiz)} quiz questions.")
    bad = [i + 1 for i, q in enumerate(quiz) if len(q.get("options") or []) != 4
           or not (0 <= int(float(q.get("answer_index", -1))) < len(q.get("options") or []))]
    if bad:
        add("invalid_quiz_items", "quiz_generator", f"Questions {bad} do not have 4 options and a valid answer.")
    notes = art.get("notes") or {}
    if len(notes.get("key_concepts") or []) < 4:
        add("thin_notes", "notes_generator", "Notes contain fewer than 4 key concepts.")
    return issues


def clean_hashtags(tags, platform: str, cap: int | None = None) -> list[str]:
    """Normalise model hashtags: '#'+CamelCase, no spaces/punctuation, de-duplicated, capped.

    Existing capitals are kept (#YOLOv8 stays #YOLOv8); multi-word tags become CamelCase.
    `cap=None` keeps everything (so the validator can still see over-long lists in the v1 baseline).
    """
    out, seen = [], set()
    if isinstance(tags, str):
        tags = [tags]
    for t in tags or []:
        for piece in re.split(r"(?=#)|,", str(t)):
            ws = [w for w in re.split(r"[^\w]+", piece) if w]
            if not ws:
                continue
            raw = ws[0] if len(ws) == 1 else "".join(w[0].upper() + w[1:] for w in ws)
            if raw.isdigit() or len(raw) < 2:
                continue
            tag = "#" + raw
            if tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return out if cap is None else out[:cap]
