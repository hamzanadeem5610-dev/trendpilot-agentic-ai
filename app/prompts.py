"""All prompts, platform rules and output schemas used by TrendPilot.

Two prompt versions are kept on purpose:
* **v1 - baseline**: the simple first-attempt prompts a beginner would write ("write a post about X").
* **v2 - improved**: the prompts after error analysis (audience + platform rules + length limits +
  few-shot examples + "never invent numbers"). The notebook runs the 10 test cases with both versions,
  so every fix in the error-analysis table is backed by a before/after comparison.
"""
from __future__ import annotations

import json

# ----------------------------------------------------------------------------------------------
# Platform rules (used by the writer prompts AND by the deterministic quality checks)
# ----------------------------------------------------------------------------------------------
PLATFORM_RULES = {
    "LinkedIn": dict(caption_words=(90, 260), max_chars=3000, hashtags=(3, 5), script_seconds=(30, 60),
                     style="Hook in the first line (it shows before 'see more'). Short paragraphs, 1-2 lines each, "
                           "a personal lesson, one concrete detail, end with a question that invites comments. "
                           "Max 2 emojis."),
    "Instagram Reels": dict(caption_words=(25, 120), max_chars=2200, hashtags=(5, 10), script_seconds=(15, 45),
                            style="Punchy first line, conversational, emojis allowed (max 4), CTA like 'save this' or 'follow for part 2'."),
    "TikTok": dict(caption_words=(8, 50), max_chars=2200, hashtags=(3, 5), script_seconds=(15, 45),
                   style="Very short caption that teases the video, casual Gen-Z friendly tone, 1-3 emojis."),
    "YouTube Shorts": dict(caption_words=(15, 80), max_chars=1000, hashtags=(2, 4), script_seconds=(20, 58),
                           style="Curiosity-driven, clear payoff, the description supports the video; title matters most."),
    "X / Twitter": dict(caption_words=(8, 45), max_chars=280, hashtags=(1, 2), script_seconds=(15, 45),
                        style="One idea, max 280 characters INCLUDING hashtags, strong opinion or surprising fact, no fluff."),
    "Facebook": dict(caption_words=(40, 150), max_chars=2000, hashtags=(1, 3), script_seconds=(20, 60),
                     style="Friendly storytelling, easy words, a question at the end to drive comments."),
}


def platform_key(platform: str) -> str:
    p = (platform or "").lower()
    if "linkedin" in p:
        return "LinkedIn"
    if "insta" in p or "reel" in p:
        return "Instagram Reels"
    if "tiktok" in p:
        return "TikTok"
    if "youtube" in p or "short" in p:
        return "YouTube Shorts"
    if "twitter" in p or p.strip() in {"x", "x / twitter", "x/twitter"} or p.startswith("x "):
        return "X / Twitter"
    if "facebook" in p or "fb" == p:
        return "Facebook"
    return "LinkedIn"


GENERIC_PHRASES = [
    "in today's fast-paced world", "in today's digital age", "game-changer", "game changer", "unlock the power",
    "unleash", "delve into", "revolutionize", "cutting-edge", "look no further", "take it to the next level",
    "dive deep", "the world of", "embark on a journey", "harness the power", "in conclusion", "elevate your",
]

# ----------------------------------------------------------------------------------------------
# JSON schemas (Ollama structured outputs)
# ----------------------------------------------------------------------------------------------
def _obj(props: dict, required: list | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props)}


S = {"type": "string"}
N = {"type": "number"}
LS = {"type": "array", "items": S}

SCHEMAS = {
    "understand": _obj({"target_audience": S, "key_message": S, "goal": S, "specific_details": LS, "avoid": LS}),
    "planner": _obj({"steps": {"type": "array", "items": _obj({"tool": S, "reason": S})}}),
    "trend_idea_generator": _obj({
        "angles": {"type": "array", "items": _obj({"title": S, "angle": S, "hook_idea": S, "why_it_works": S, "score": N})},
        "best_index": N, "target_audience": S}),
    "caption_writer": _obj({"hook": S, "caption": S, "call_to_action": S}),
    "hashtag_generator": _obj({"hashtags": LS}),
    "reel_script_generator": _obj({
        "total_seconds": N,
        "scenes": {"type": "array", "items": _obj({"time": S, "visual": S, "voiceover": S, "on_screen_text": S})},
        "closing_line": S}),
    "title_thumbnail_generator": _obj({"titles": LS, "best_title": S, "thumbnail_text": S}),
    "image_prompt_generator": _obj({"image_prompt": S, "alt_text": S}),
    "content_reviewer": _obj({
        "scores": _obj({"hook_strength": N, "clarity": N, "length_fit": N, "tone_fit": N,
                        "hashtag_relevance": N, "platform_readiness": N}),
        "overall": N, "verdict": S, "issues": LS, "suggestions": LS, "rewrite": LS}),
    "feedback_router": _obj({"tools": LS, "instruction": S}),
    # ---- StudyBuddy ----
    "study_planner": _obj({
        "title": S,
        "days": {"type": "array", "items": _obj({
            "day": N, "focus": S, "goals": LS, "tasks": LS, "minutes": N,
            "resources": {"type": "array", "items": _obj({"name": S, "type": S})}})}}),
    "notes_generator": _obj({
        "summary": S,
        "key_concepts": {"type": "array", "items": _obj({"term": S, "explanation": S})},
        "cheat_sheet": LS, "common_mistakes": LS}),
    "quiz_generator": _obj({"questions": {"type": "array", "items": _obj({
        "question": S, "options": LS, "answer_index": N, "explanation": S, "difficulty": S})}}),
    "study_reviewer": _obj({
        "scores": _obj({"coverage": N, "pacing": N, "clarity": N, "quiz_quality": N}),
        "overall": N, "verdict": S, "issues": LS, "suggestions": LS, "rewrite": LS}),
}

# ----------------------------------------------------------------------------------------------
# System prompts
# ----------------------------------------------------------------------------------------------
SYSTEM_V1 = "You are a helpful assistant that writes social media content. Reply in JSON."

SYSTEM_V2 = (
    "You are TrendPilot, a senior social-media strategist for tech creators and students. "
    "You write specific, human, platform-native content - never generic corporate filler. "
    "Rules: (1) Use concrete details from the topic. (2) NEVER invent numbers, metrics, companies or quotes; "
    "if a number would help, write a placeholder in square brackets like [your mAP score]. "
    "(3) Respect the platform rules and length limits exactly. (4) Avoid cliches such as: "
    + ", ".join(GENERIC_PHRASES[:10]) + ". (5) Reply with ONE valid JSON object only - no markdown, no commentary."
)

SYSTEM_STUDY = (
    "You are StudyBuddy, an expert tutor who designs realistic study plans for university students. "
    "Be specific and practical. Do not invent URLs - name resources (official docs, well-known courses, books) "
    "so the student can search for them. Reply with ONE valid JSON object only."
)

FEW_SHOT_HOOKS = """Examples of strong hooks (style reference only - do not copy):
- "I trained an AI to spot missing helmets. The hardest part wasn't the model - it was the data."
- "My model hit 0.9 mAP on validation. Then I tested it on a real photo."
- "3 things nobody tells you about your first ML internship:"
Examples of weak hooks (avoid): "Excited to share my new project!", "In today's world, AI is everywhere." """

FEW_SHOT_HUMOR = """Humor guide (for funny / meme / humorous tones): use a relatable developer pain point,
a setup -> twist structure, self-deprecation, and familiar formats ("Nobody: / Me:", "POV:", "Expectation vs reality").
Example: "POV: your model gets 98% accuracy... on the training set. Production: 'I've never seen this data in my life.'" """


def _ctx(req: dict, extra: dict | None = None) -> str:
    rules = PLATFORM_RULES[platform_key(req["platform"])]
    lines = [f"Topic: {req['topic']}", f"Platform: {platform_key(req['platform'])}", f"Tone: {req['tone']}",
             f"Requested output: {req['output_type']}"]
    for k, v in (extra or {}).items():
        if v:
            lines.append(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}")
    lines.append(f"Platform style: {rules['style']}")
    return "\n".join(lines)


def _feedback(fb: str | None) -> str:
    return f"\n\nIMPORTANT - improve the previous version using this reviewer/user feedback:\n{fb}" if fb else ""


# ----------------------------------------------------------------------------------------------
# Prompt builders (one per tool). `v` = "v1" or "v2".
# ----------------------------------------------------------------------------------------------
def understand_prompt(req, memory_notes, research, v):
    if v == "v1":
        return f"Describe the audience and goal for a {req['platform']} post about: {req['topic']}"
    return (_ctx(req, {"What we already did before (avoid repeating)": memory_notes,
                       "Fresh web results about this topic": research}) +
            "\n\nAnalyse the user's goal. Return: target_audience (who exactly will read it), key_message "
            "(one sentence), goal (e.g. get recruiter attention, educate, entertain), specific_details "
            "(3-5 concrete facts taken ONLY from the topic text - no invented numbers), avoid (2-4 things that "
            "would make this post weak or off-brand).")


def planner_prompt(req, tool_catalog: dict, understanding):
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in tool_catalog.items())
    return (_ctx(req, {"Goal analysis": understanding}) +
            f"\n\nYou are the PLANNER agent. Available tools:\n{catalog}\n\n"
            "Create an ordered plan that produces exactly what the user requested. Use only tool names from the "
            "list. Start with idea generation, put the reviewer near the end and the file_saver last. "
            "Return steps as [{\"tool\": name, \"reason\": short reason}].")


def idea_prompt(req, ctx, v):
    if v == "v1":
        return f"Give 5 viral content ideas about {req['topic']} for {req['platform']}. Pick the best one."
    return (_ctx(req, ctx) + "\n\n" + FEW_SHOT_HOOKS +
            "\n\nGenerate 5 DIFFERENT viral content angles (e.g. lesson learned, behind-the-scenes, mistake/failure, "
            "how-to, contrarian take). For each give: title, angle, hook_idea (max 20 words), why_it_works, "
            "score 1-10 for this audience and platform. Set best_index (0-4) to the strongest and describe the "
            "target_audience in one line.")


def caption_prompt(req, ctx, feedback, v):
    if v == "v1":
        return f"Write a {req['tone']} {req['platform']} post about {req['topic']}. Give hook, caption and call_to_action." + _feedback(feedback)
    r = PLATFORM_RULES[platform_key(req["platform"])]
    lo, hi = r["caption_words"]
    extra = FEW_SHOT_HUMOR if any(t in req["tone"].lower() for t in ("fun", "humor", "meme")) else FEW_SHOT_HOOKS
    return (_ctx(req, ctx) + "\n\n" + extra +
            f"\n\nWrite the post. hook = the first line (max 20 words, must create curiosity). caption = the FULL "
            f"post text starting with the hook, {lo}-{hi} words, max {r['max_chars']} characters, short lines, "
            f"in a '{req['tone']}' tone. Do NOT put hashtags in the caption (a separate tool adds them). "
            "call_to_action = the closing line inviting engagement." + _feedback(feedback))


def hashtag_prompt(req, ctx, feedback, v):
    lo, hi = PLATFORM_RULES[platform_key(req["platform"])]["hashtags"]
    if v == "v1":
        return f"Generate hashtags for a post about {req['topic']}." + _feedback(feedback)
    return (_ctx(req, ctx) +
            f"\n\nGenerate {lo}-{hi} hashtags for this exact post. Mix: 1-2 broad (e.g. #AI), the rest niche and "
            "specific to the topic. Every hashtag must be relevant to the topic - no generic spam like #love "
            "#instagood #viral #trending. Format: '#' + CamelCase words, no spaces." + _feedback(feedback))


def reel_prompt(req, ctx, feedback, v):
    if v == "v1":
        return f"Write a reel script about {req['topic']} for {req['platform']}." + _feedback(feedback)
    lo, hi = PLATFORM_RULES[platform_key(req["platform"])]["script_seconds"]
    target = min(hi, max(lo, 30))
    return (_ctx(req, ctx) +
            f"\n\nWrite a vertical short-video script of about {target} seconds (strictly between {lo} and {hi}). "
            "4-6 scenes. Scene 1 (0-3s) must be the opening hook. Then main explanation, then a payoff. "
            "Each scene: time (like '0-3s'), visual (what is on screen), voiceover (what is said - max 25 words), "
            "on_screen_text (max 6 words). total_seconds = the end time of the last scene. closing_line = the "
            "final line that drives follows/comments." + _feedback(feedback))


def title_prompt(req, ctx, feedback, v):
    if v == "v1":
        return f"Give titles and thumbnail text for a video about {req['topic']}." + _feedback(feedback)
    return (_ctx(req, ctx) +
            "\n\nGenerate 3 catchy titles (max 60 characters each, curiosity + clear payoff, no clickbait lies), "
            "pick best_title, and thumbnail_text (max 5 words, big-and-bold style)." + _feedback(feedback))


def image_prompt(req, ctx, feedback, v):
    return (_ctx(req, ctx) +
            "\n\nWrite one detailed prompt for an AI image generator to create the post's cover image "
            "(subject, composition, style, colours, 16:9 or 4:5). No text in the image. Also give short alt_text."
            + _feedback(feedback))


def reviewer_prompt(req, content: dict, rule_issues: list, v):
    return (_ctx(req) + "\n\nYou are the REVIEWER agent. Review this content critically:\n"
            + json.dumps(content, ensure_ascii=False, indent=1)
            + (f"\n\nAutomatic checks already found these problems: {rule_issues}" if rule_issues else "")
            + "\n\nScore 1-10 for: hook_strength, clarity, length_fit (fits platform limits), tone_fit, "
              "hashtag_relevance, platform_readiness. overall = average. verdict = 'ready' if overall >= 7.5 "
              "and there are no serious problems, else 'needs_revision'. issues = concrete problems. "
              "suggestions = concrete fixes. rewrite = list of tool names that must be re-run, chosen from: "
              "caption_writer, hashtag_generator, reel_script_generator, title_thumbnail_generator. "
              "Be strict - generic or overly long content must not get more than 6.")


def feedback_router_prompt(req, feedback: str, tools_used: list):
    return (_ctx(req) + f"\n\nThe user gave this feedback on the generated content: \"{feedback}\"\n"
            f"Tools that produced the content: {tools_used}.\n"
            "Decide which tools must be re-run to apply the feedback (tools = list of tool names from the list "
            "above) and write a clear instruction for them.")


# ---- StudyBuddy ----
def study_ctx(req: dict) -> str:
    return (f"Topic: {req['topic']}\nDays available: {req['days']}\nStudent level: {req['level']}\n"
            f"Time per day: {req['hours_per_day']} hours\nGoal: {req.get('goal') or 'understand and apply the topic'}")


def study_plan_prompt(req, memory_notes, feedback):
    return (study_ctx(req) + (f"\nPrevious study sessions: {memory_notes}" if memory_notes else "") +
            f"\n\nCreate a {req['days']}-day study plan. For each day: day number, focus, 2-4 goals, 3-5 concrete "
            "tasks (read / watch / code / practise), minutes (must fit the daily time), and 1-3 resources "
            "(name + type like docs/video/course/book/practice). Build from basics to a small hands-on project on "
            "the last day. title = a motivating plan title." + _feedback(feedback))


def notes_prompt(req, plan_titles, feedback):
    return (study_ctx(req) + f"\nPlan focus areas: {plan_titles}\n\nWrite revision notes: summary (5-7 sentences), "
            "6-10 key_concepts (term + simple explanation), cheat_sheet (6-10 one-line facts/commands), "
            "common_mistakes (3-5)." + _feedback(feedback))


def quiz_prompt(req, concepts, feedback):
    return (study_ctx(req) + f"\nKey concepts: {concepts}\n\nCreate 8 multiple-choice questions that test "
            "understanding (not trivia). Each: question, exactly 4 options, answer_index (0-3), a short explanation, "
            "difficulty (easy/medium/hard). Mix difficulties and make wrong options plausible." + _feedback(feedback))


def study_reviewer_prompt(req, content: dict, rule_issues: list):
    return (study_ctx(req) + "\n\nYou are the REVIEWER agent. Review this study pack:\n"
            + json.dumps(content, ensure_ascii=False)[:6000]
            + (f"\n\nAutomatic checks found: {rule_issues}" if rule_issues else "")
            + "\n\nScore 1-10: coverage, pacing (fits days and time), clarity, quiz_quality. overall = average. "
              "verdict 'ready' or 'needs_revision'. issues, suggestions, rewrite = tools to re-run from: "
              "study_planner, notes_generator, quiz_generator.")
