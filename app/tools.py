"""Agent tools.

Every tool is a small Python function registered in `REGISTRY` with a name and a description. The planner
agent sees the descriptions and decides which tools to call; the executor calls them and logs
`[Tool Called]: <name>`. Tools read and write a shared blackboard (`ctx.artifacts`).

TrendPilot tools (8 of the brief's tools + 2 extras):
  memory_tool, trend_research (web), trend_idea_generator, caption_writer, hashtag_generator,
  reel_script_generator, title_thumbnail_generator, image_prompt_generator, content_reviewer, file_saver
StudyBuddy tools (brief's Option B list):
  study_planner, notes_generator, quiz_generator, study_reviewer, progress_saver (+ file saving)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts as P
from .config import Settings
from .llm import OllamaClient
from .memory import Memory
from .quality import check_content, check_study, clean_hashtags, core_topic, parts_for, script_seconds, topic_keywords


@dataclass
class ToolContext:
    llm: OllamaClient
    settings: Settings
    memory: Memory
    request: dict
    artifacts: dict = field(default_factory=dict)
    feedback: dict = field(default_factory=dict)      # tool name -> improvement instruction (revision rounds)

    @property
    def v(self) -> str:
        return self.settings.prompt_version

    @property
    def system(self) -> str:
        return P.SYSTEM_V2 if self.v == "v2" else P.SYSTEM_V1

    def writer_context(self) -> dict:
        """Facts the writer tools share (v2 only - the v1 baseline prompts ignore them)."""
        a = self.artifacts
        u = a.get("understanding", {})
        idea = a.get("content_idea") or {}
        return {
            "Target audience": a.get("target_audience") or u.get("target_audience"),
            "Key message": u.get("key_message"),
            "Concrete details to use": u.get("specific_details"),
            "Avoid": u.get("avoid"),
            "Chosen content angle": f"{idea.get('title', '')} - {idea.get('angle', '')} (hook idea: {idea.get('hook_idea', '')})" if idea else "",
            "Previous work (memory)": a.get("memory_notes"),
            "Web research": a.get("research"),
        }


@dataclass
class Tool:
    name: str
    display_name: str
    description: str
    mode: str                    # "content" | "study" | "shared"
    fn: Callable
    uses_llm: bool = True

    def run(self, ctx: ToolContext) -> dict:
        return self.fn(ctx)


REGISTRY: dict[str, Tool] = {}


def tool(name, display_name, description, mode, uses_llm=True):
    def deco(fn):
        REGISTRY[name] = Tool(name, display_name, description, mode, fn, uses_llm)
        return fn
    return deco


def catalog(mode: str) -> dict:
    """Name -> description of the tools the planner may choose for a mode."""
    return {t.name: t.description for t in REGISTRY.values() if t.mode in (mode, "shared")}


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:n].rstrip("_") or "content"


# ==============================================================================================
# Shared tools
# ==============================================================================================
@tool("memory_tool", "Simple Memory Tool",
      "Recall previous topics, platforms, styles and hooks from long-term memory (SQLite) so the new content is fresh.",
      "shared", uses_llm=False)
def memory_tool(ctx: ToolContext) -> dict:
    req = ctx.request
    if "days" in req:                                   # StudyBuddy request
        notes = ctx.memory.recall_study(req["topic"])
        ctx.artifacts["memory_notes"] = notes
        return {"recalled": notes or "no related study plans"}
    hits = ctx.memory.recall(req["topic"], req.get("platform"))
    ctx.artifacts["memory_hits"] = [{k: h[k] for k in ("topic", "platform", "tone", "hook", "similarity")} for h in hits]
    ctx.artifacts["memory_notes"] = ctx.memory.memory_notes(req["topic"], req.get("platform"))
    return {"recalled_runs": len(hits), "preferences": ctx.memory.prefs()}


# ==============================================================================================
# TrendPilot tools
# ==============================================================================================
@tool("trend_research", "Web Trend Research (DuckDuckGo)",
      "Search the web (DuckDuckGo) for fresh, real context about the topic so the content is current and factual.",
      "content", uses_llm=False)
def trend_research(ctx: ToolContext) -> dict:
    if not ctx.settings.enable_web_search or ctx.v == "v1":
        ctx.artifacts["research"] = ""
        return {"skipped": "web search disabled"}
    q = f"{ctx.request['topic']} {P.platform_key(ctx.request['platform'])} tips"
    try:
        from ddgs import DDGS
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(1) as ex:
            fut = ex.submit(lambda: list(DDGS().text(q, max_results=5)))
            results = fut.result(timeout=ctx.settings.web_search_timeout)
    except Exception as e:                           # offline / rate-limited -> the agent continues without it
        ctx.artifacts["research"] = ""
        return {"skipped": f"search unavailable ({type(e).__name__})"}
    snippets = [f"{r.get('title', '')}: {r.get('body', '')[:160]}" for r in results]
    ctx.artifacts["research"] = " || ".join(snippets)[:900]
    ctx.artifacts["research_sources"] = [r.get("href") for r in results if r.get("href")]
    return {"results": len(results), "query": q}


@tool("trend_idea_generator", "Trend Idea Generator",
      "Generate 5 viral content angles for the topic and platform, score them and select the best one.", "content")
def trend_idea_generator(ctx: ToolContext) -> dict:
    extra = ctx.writer_context()
    data = ctx.llm.generate_json(P.idea_prompt(ctx.request, extra, ctx.v), ctx.system, P.SCHEMAS["trend_idea_generator"],
                                 tool="trend_idea_generator", temperature=0.9)
    angles = data.get("angles") or []
    try:
        best = int(float(data.get("best_index", 0)))
    except (TypeError, ValueError):
        best = 0
    if angles and not 0 <= best < len(angles):
        best = max(range(len(angles)), key=lambda i: float(angles[i].get("score") or 0))
    ctx.artifacts["ideas"] = angles
    ctx.artifacts["content_idea"] = angles[best] if angles else {}
    if data.get("target_audience"):
        ctx.artifacts["target_audience"] = data["target_audience"]
    return {"angles": len(angles), "selected": (angles[best].get("title") if angles else None)}


def fit_chars(text: str, limit: int) -> str:
    """Cut text to whole sentences (or words) so it fits `limit` characters."""
    if len(text) <= limit:
        return text
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = ""
    for part in parts:
        cand = (out + " " + part).strip() if out else part.strip()
        if len(cand) > limit:
            break
        out = cand
    if not out:
        out = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return out


@tool("caption_writer", "Caption Writer",
      "Write the hook, the full platform-ready post caption and a call to action in the requested tone.", "content")
def caption_writer(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.caption_prompt(ctx.request, ctx.writer_context(), ctx.feedback.get("caption_writer"), ctx.v),
                                 ctx.system, P.SCHEMAS["caption_writer"], tool="caption_writer", max_tokens=1100)
    hook = (data.get("hook") or "").strip()
    caption = (data.get("caption") or "").strip()
    cta = (data.get("call_to_action") or "").strip()
    if ctx.v == "v2":
        if hook and not caption.lower().startswith(hook[:25].lower()):
            caption = f"{hook}\n\n{caption}"                     # the hook must be the first line
        if cta and cta[:30].lower() not in caption.lower():
            caption = f"{caption}\n\n{cta}"
        pk = P.platform_key(ctx.request["platform"])
        if pk != "X / Twitter":                                    # hashtags belong to the hashtag tool
            caption = re.sub(r"(\s#\w+)+\s*$", "", caption).rstrip()
        limit = P.PLATFORM_RULES[pk]["max_chars"]
        budget = limit - 30 if limit <= 500 else limit             # leave room for hashtags on X
        if len(caption) > budget:                                  # hard limit: one LLM shorten, then trim
            try:
                short = ctx.llm.generate_json(
                    P.caption_prompt(ctx.request, ctx.writer_context(),
                                     f"The caption is {len(caption)} characters. Rewrite it in UNDER {budget} characters "
                                     f"(one or two short sentences). Keep the hook idea.", ctx.v),
                    ctx.system, P.SCHEMAS["caption_writer"], tool="caption_writer", max_tokens=400)
                if short.get("caption") and len(short["caption"]) < len(caption):
                    caption = short["caption"].strip()
                    hook = (short.get("hook") or hook).strip()
            except Exception:
                pass
            caption = fit_chars(caption, budget)
    ctx.artifacts.update(hook=hook, caption=caption, call_to_action=cta)
    return {"hook_words": len(hook.split()), "caption_words": len(caption.split())}


@tool("hashtag_generator", "Hashtag Generator",
      "Generate relevant hashtags for the platform (right count, broad + niche, no spam tags).", "content")
def hashtag_generator(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.hashtag_prompt(ctx.request, ctx.writer_context(), ctx.feedback.get("hashtag_generator"), ctx.v),
                                 ctx.system, P.SCHEMAS["hashtag_generator"], tool="hashtag_generator", temperature=0.5, max_tokens=300)
    cap = P.PLATFORM_RULES[P.platform_key(ctx.request["platform"])]["hashtags"][1] if ctx.v == "v2" else None
    raw = [t for t in (data.get("hashtags") or []) if isinstance(t, str)]
    if ctx.v == "v2":                                              # topic-specific tags first, so the cap keeps them
        kw = topic_keywords(ctx.request.get("core_topic") or ctx.request["topic"])
        raw.sort(key=lambda t: not any(k in t.lower() for k in kw))
        if kw and not any(k in " ".join(raw).lower() for k in kw):
            raw.insert(0, "#" + "".join(w.title() for w in sorted(kw, key=len, reverse=True)[:2]))
    tags = clean_hashtags(raw, ctx.request["platform"], cap=cap)
    ctx.artifacts["hashtags"] = tags
    return {"hashtags": len(tags)}


@tool("reel_script_generator", "Reel Script Generator",
      "Create a 15-60 second short-video script: opening hook, explanation, visual suggestions and closing line.", "content")
def reel_script_generator(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.reel_prompt(ctx.request, ctx.writer_context(), ctx.feedback.get("reel_script_generator"), ctx.v),
                                 ctx.system, P.SCHEMAS["reel_script_generator"], tool="reel_script_generator", max_tokens=1200)
    scenes = [s for s in (data.get("scenes") or []) if isinstance(s, dict)]
    script = {"total_seconds": data.get("total_seconds"), "scenes": scenes, "closing_line": data.get("closing_line", "")}
    script["total_seconds"] = script_seconds(script)
    ctx.artifacts["reel_script"] = script
    return {"scenes": len(scenes), "seconds": script["total_seconds"]}


@tool("title_thumbnail_generator", "Title and Thumbnail Text Generator",
      "Generate catchy video/post titles and short thumbnail text.", "content")
def title_thumbnail_generator(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.title_prompt(ctx.request, ctx.writer_context(), ctx.feedback.get("title_thumbnail_generator"), ctx.v),
                                 ctx.system, P.SCHEMAS["title_thumbnail_generator"], tool="title_thumbnail_generator", max_tokens=300)
    titles = [t for t in (data.get("titles") or []) if isinstance(t, str)]
    best = data.get("best_title") or (titles[0] if titles else "")
    ctx.artifacts["titles"] = {"titles": titles, "best_title": best, "thumbnail_text": data.get("thumbnail_text", "")}
    return {"best_title": best}


@tool("image_prompt_generator", "Post Image Prompt Generator",
      "Write a prompt for an AI image generator to create the post's cover image (optional extra).", "content")
def image_prompt_generator(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.image_prompt(ctx.request, ctx.writer_context(), ctx.feedback.get("image_prompt_generator"), ctx.v),
                                 ctx.system, P.SCHEMAS["image_prompt_generator"], tool="image_prompt_generator", max_tokens=300)
    ctx.artifacts["image_prompt"] = {"prompt": data.get("image_prompt", ""), "alt_text": data.get("alt_text", "")}
    return {"chars": len(data.get("image_prompt", ""))}


def content_view(art: dict) -> dict:
    """The part of the blackboard the reviewer sees."""
    keys = ("hook", "caption", "hashtags", "reel_script", "titles")
    return {k: art[k] for k in keys if art.get(k)}


@tool("content_reviewer", "Content Reviewer",
      "Review the content (hook strength, clarity, length, tone, hashtag relevance, platform readiness) and decide what to improve.",
      "content")
def content_reviewer(ctx: ToolContext) -> dict:
    rule_issues = check_content(ctx.request, ctx.artifacts)
    llm = ctx.llm.generate_json(P.reviewer_prompt(ctx.request, content_view(ctx.artifacts), [i["message"] for i in rule_issues], ctx.v),
                                ctx.system, P.SCHEMAS["content_reviewer"], tool="content_reviewer", temperature=0.2, max_tokens=700)
    scores = {k: _num(v) for k, v in (llm.get("scores") or {}).items()}
    llm_overall = _num(llm.get("overall")) or (sum(scores.values()) / len(scores) if scores else 5.0)
    majors = [i for i in rule_issues if i["severity"] == "major"]
    minors = [i for i in rule_issues if i["severity"] != "major"]
    final = max(0.0, min(10.0, llm_overall - 0.75 * len(majors) - 0.25 * len(minors)))
    needed = parts_for(ctx.request["output_type"], ctx.request["platform"])
    produced_by = {"caption": "caption_writer", "hashtags": "hashtag_generator", "reel_script": "reel_script_generator",
                   "titles": "title_thumbnail_generator"}
    allowed = {produced_by[p] for p in needed if p in produced_by}
    rewrite = sorted({t for t in (llm.get("rewrite") or []) if t in allowed} | {i["tool"] for i in majors if i["tool"] in allowed})
    verdict = "ready" if final >= ctx.settings.review_threshold and not majors else "needs_revision"
    review = {"scores": scores, "llm_overall": round(llm_overall, 2), "overall": round(final, 2), "verdict": verdict,
              "issues": [i["message"] for i in rule_issues] + [x for x in (llm.get("issues") or []) if isinstance(x, str)],
              "suggestions": [x for x in (llm.get("suggestions") or []) if isinstance(x, str)],
              "rule_issues": rule_issues, "rewrite": rewrite if verdict == "needs_revision" else []}
    ctx.artifacts["review"] = review
    return {"overall": review["overall"], "verdict": verdict, "rule_issues": len(rule_issues), "rewrite": review["rewrite"]}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------------- File saver
def render_markdown(result: dict) -> str:
    req, a = result["request"], result["artifacts"]
    rev = a.get("review", {})
    L = [f"# {req['topic']}", "",
         f"**Platform:** {req['platform']}  |  **Tone:** {req['tone']}  |  **Output:** {req['output_type']}  ",
         f"**Generated:** {result.get('created_at', '')}  |  **Model:** {result.get('model', '')}  |  **Prompt version:** {result.get('prompt_version', '')}", ""]
    idea = a.get("content_idea") or {}
    if idea:
        L += ["## Content Idea", f"**{idea.get('title', '')}** - {idea.get('angle', '')}", "",
              f"*Why it works:* {idea.get('why_it_works', '')}", ""]
    if a.get("target_audience"):
        L += ["## Target Audience", a["target_audience"], ""]
    if a.get("hook"):
        L += ["## Viral Hook", f"> {a['hook']}", ""]
    if a.get("caption"):
        L += ["## Post Caption", a["caption"], ""]
    if a.get("hashtags"):
        L += ["## Hashtags", " ".join(a["hashtags"]), ""]
    sc = a.get("reel_script")
    if sc and sc.get("scenes"):
        L += [f"## Reel / Short Video Script (~{sc.get('total_seconds', 0):.0f}s)", "",
              "| Time | Visual | Voiceover | On-screen text |", "|---|---|---|---|"]
        for s in sc["scenes"]:
            L.append("| {} | {} | {} | {} |".format(*(str(s.get(k, "")).replace("|", "/").replace("\n", " ")
                                                       for k in ("time", "visual", "voiceover", "on_screen_text"))))
        L += ["", f"**Closing line:** {sc.get('closing_line', '')}", ""]
    t = a.get("titles")
    if t and t.get("best_title"):
        L += ["## Suggested Title", f"**{t['best_title']}**", ""]
        others = [x for x in t.get("titles", []) if x != t["best_title"]]
        if others:
            L += ["Other options: " + " / ".join(others), ""]
        L += ["## Thumbnail Text", f"**{t.get('thumbnail_text', '')}**", ""]
    if a.get("image_prompt", {}).get("prompt"):
        L += ["## Cover Image Prompt", a["image_prompt"]["prompt"], ""]
    if rev:
        L += ["## Content Review & Improvement Suggestions",
              f"**Score:** {rev.get('overall')} / 10  |  **Verdict:** {rev.get('verdict')}  |  **Revisions made:** {result.get('revisions', 0)}", ""]
        if rev.get("scores"):
            L += [", ".join(f"{k.replace('_', ' ')}: {v:g}" for k, v in rev["scores"].items()), ""]
        L += [f"- {s}" for s in rev.get("suggestions", [])[:6]] + [""]
        if rev.get("issues"):
            L += ["Remaining issues:"] + [f"- {s}" for s in rev["issues"][:6]] + [""]
    if a.get("research_sources"):
        L += ["## Web Sources Checked"] + [f"- {u}" for u in a["research_sources"][:5]] + [""]
    L += ["## Agent Trace", "Tools called: " + " -> ".join(result.get("tools_used", [])), ""]
    return "\n".join(L)


@tool("file_saver", "File Saver Tool",
      "Save the final result as Markdown + JSON in outputs/ (posts, scripts, full results). Mandatory last step.",
      "shared", uses_llm=False)
def file_saver(ctx: ToolContext) -> dict:
    result = ctx.artifacts["_result_snapshot"]
    s = ctx.settings
    s.ensure_dirs()
    req = ctx.request
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if "days" in req:
        base = f"{slug(req['topic'])}_{req['days']}days_{stamp}"
    else:
        base = f"{slug(req.get('core_topic') or req['topic'])}_{slug(P.platform_key(req['platform']), 12)}_{stamp}"
    if result.get("user_feedback") or ctx.artifacts.get("_is_revision"):
        base += "_revised"
    n, root = 2, base                                   # never overwrite an earlier result
    while any(next(d.glob(f"*{base}*"), None) for d in (s.posts_dir, s.scripts_dir, s.results_dir, s.study_dir)):
        base, n = f"{root}_{n}", n + 1
    saved = {}
    if "days" in req:                               # StudyBuddy pack
        md_path = s.study_dir / f"study_{base}.md"
        md_path.write_text(render_study_markdown(result), encoding="utf-8")
        saved["markdown"] = str(md_path)
    else:
        md = render_markdown(result)
        a = ctx.artifacts
        if a.get("caption"):
            p = s.posts_dir / f"{base}_post.md"
            p.write_text(md, encoding="utf-8")
            saved["markdown"] = str(p)
        if a.get("reel_script", {}).get("scenes"):
            p = s.scripts_dir / f"{base}_script.md"
            p.write_text(md if not a.get("caption") else render_script_only(result), encoding="utf-8")
            saved["script"] = str(p)
            saved.setdefault("markdown", str(p))
        if not saved:
            p = s.posts_dir / f"{base}.md"
            p.write_text(md, encoding="utf-8")
            saved["markdown"] = str(p)
    js = s.results_dir / f"{'study_' if 'days' in req else ''}{base}.json"
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    clean["artifacts"] = {k: v for k, v in result["artifacts"].items() if not k.startswith("_")}
    clean["saved_files"] = {**saved, "json": str(js)}
    clean["artifacts"]["saved_files"] = clean["saved_files"]
    from .agent import TrendPilotAgent                   # lazy import (agent imports this module)
    clean["final_output"] = TrendPilotAgent.final_output(clean)
    js.write_text(json.dumps(clean, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    saved["json"] = str(js)
    ctx.artifacts["saved_files"] = saved
    return {k: Path(v).name for k, v in saved.items()}


def render_script_only(result: dict) -> str:
    req, sc = result["request"], result["artifacts"].get("reel_script", {})
    t = result["artifacts"].get("titles", {})
    L = [f"# Reel Script - {req['topic']}", f"Platform: {req['platform']} | Tone: {req['tone']} | ~{sc.get('total_seconds', 0):.0f}s", ""]
    if t.get("best_title"):
        L += [f"**Title:** {t['best_title']}  |  **Thumbnail:** {t.get('thumbnail_text', '')}", ""]
    for i, s in enumerate(sc.get("scenes", []), 1):
        L += [f"**Scene {i} ({s.get('time', '')})**", f"- Visual: {s.get('visual', '')}", f"- Voiceover: \"{s.get('voiceover', '')}\"",
              f"- On-screen text: {s.get('on_screen_text', '')}", ""]
    L.append(f"**Closing line:** {sc.get('closing_line', '')}")
    return "\n".join(L)


# ==============================================================================================
# StudyBuddy tools (Option B)
# ==============================================================================================
@tool("study_planner", "Study Planner",
      "Create a realistic day-by-day study plan with goals, tasks, time and resources.", "study")
def study_planner(ctx: ToolContext) -> dict:
    data = ctx.llm.generate_json(P.study_plan_prompt(ctx.request, ctx.artifacts.get("memory_notes"), ctx.feedback.get("study_planner")),
                                 P.SYSTEM_STUDY, P.SCHEMAS["study_planner"], tool="study_planner", max_tokens=2200)
    days = [d for d in (data.get("days") or []) if isinstance(d, dict)]
    for i, d in enumerate(days, 1):
        d["day"] = int(_num(d.get("day")) or i)
    ctx.artifacts["study_plan"] = {"title": data.get("title", ""), "days": days}
    return {"days": len(days)}


@tool("notes_generator", "Notes Generator",
      "Write revision notes: summary, key concepts, cheat sheet and common mistakes.", "study")
def notes_generator(ctx: ToolContext) -> dict:
    focus = [d.get("focus") for d in ctx.artifacts.get("study_plan", {}).get("days", [])]
    data = ctx.llm.generate_json(P.notes_prompt(ctx.request, focus, ctx.feedback.get("notes_generator")),
                                 P.SYSTEM_STUDY, P.SCHEMAS["notes_generator"], tool="notes_generator", max_tokens=1600)
    ctx.artifacts["notes"] = data
    return {"concepts": len(data.get("key_concepts") or [])}


@tool("quiz_generator", "Quiz Generator",
      "Create multiple-choice quiz questions with answers and explanations.", "study")
def quiz_generator(ctx: ToolContext) -> dict:
    concepts = [c.get("term") for c in (ctx.artifacts.get("notes", {}).get("key_concepts") or []) if isinstance(c, dict)]
    data = ctx.llm.generate_json(P.quiz_prompt(ctx.request, concepts, ctx.feedback.get("quiz_generator")),
                                 P.SYSTEM_STUDY, P.SCHEMAS["quiz_generator"], tool="quiz_generator", max_tokens=1800)
    qs = []
    for q in data.get("questions") or []:
        if isinstance(q, dict) and len(q.get("options") or []) >= 2:
            q["answer_index"] = int(_num(q.get("answer_index")))
            qs.append(q)
    ctx.artifacts["quiz"] = {"questions": qs}
    return {"questions": len(qs)}


@tool("study_reviewer", "Study Pack Reviewer",
      "Review the plan, notes and quiz (coverage, pacing, clarity, quiz quality) and decide what to improve.", "study")
def study_reviewer(ctx: ToolContext) -> dict:
    rule_issues = check_study(ctx.request, ctx.artifacts)
    view = {k: ctx.artifacts.get(k) for k in ("study_plan", "notes", "quiz")}
    llm = ctx.llm.generate_json(P.study_reviewer_prompt(ctx.request, view, [i["message"] for i in rule_issues]),
                                P.SYSTEM_STUDY, P.SCHEMAS["study_reviewer"], tool="study_reviewer", temperature=0.2, max_tokens=600)
    scores = {k: _num(v) for k, v in (llm.get("scores") or {}).items()}
    llm_overall = _num(llm.get("overall")) or (sum(scores.values()) / len(scores) if scores else 5.0)
    majors = [i for i in rule_issues if i["severity"] == "major"]
    final = max(0.0, min(10.0, llm_overall - 0.75 * len(majors)))
    allowed = {"study_planner", "notes_generator", "quiz_generator"}
    rewrite = sorted({t for t in (llm.get("rewrite") or []) if t in allowed} | {i["tool"] for i in majors})
    verdict = "ready" if final >= ctx.settings.review_threshold and not majors else "needs_revision"
    ctx.artifacts["review"] = {"scores": scores, "llm_overall": round(llm_overall, 2), "overall": round(final, 2),
                               "verdict": verdict, "issues": [i["message"] for i in rule_issues] + list(llm.get("issues") or []),
                               "suggestions": list(llm.get("suggestions") or []), "rule_issues": rule_issues,
                               "rewrite": rewrite if verdict == "needs_revision" else []}
    return {"overall": round(final, 2), "verdict": verdict}


@tool("progress_saver", "Progress Saver",
      "Save the study plan and a day-by-day progress tracker to memory (SQLite) so progress can be updated later.",
      "study", uses_llm=False)
def progress_saver(ctx: ToolContext) -> dict:
    plan = ctx.artifacts.get("study_plan", {})
    pid = ctx.memory.save_study_plan(ctx.request, plan.get("title", ""), plan.get("days", []), "")
    ctx.artifacts["plan_id"] = pid
    ctx.artifacts["progress"] = ctx.memory.progress(pid)
    return {"plan_id": pid, "days_tracked": ctx.artifacts["progress"]["days_total"]}


def render_study_markdown(result: dict) -> str:
    req, a = result["request"], result["artifacts"]
    plan, notes, quiz, rev = a.get("study_plan", {}), a.get("notes", {}), a.get("quiz", {}), a.get("review", {})
    L = [f"# {plan.get('title') or req['topic']}", "",
         f"**Topic:** {req['topic']}  |  **Days:** {req['days']}  |  **Level:** {req['level']}  |  **Time/day:** {req['hours_per_day']} h  ",
         f"**Plan ID (progress tracking):** {a.get('plan_id', '-')}  |  **Generated:** {result.get('created_at', '')}", "",
         "## Daily Study Plan"]
    for d in plan.get("days", []):
        L += [f"### Day {d.get('day')}: {d.get('focus', '')} ({_num(d.get('minutes')):.0f} min)"]
        L += [f"- Goal: {g}" for g in d.get("goals", [])] + [f"- [ ] {t}" for t in d.get("tasks", [])]
        res = d.get("resources") or []
        if res:
            L.append("- Resources: " + "; ".join(f"{r.get('name')} ({r.get('type')})" for r in res if isinstance(r, dict)))
        L.append("")
    if notes:
        L += ["## Revision Notes", notes.get("summary", ""), "", "### Key Concepts"]
        L += [f"- **{c.get('term')}** - {c.get('explanation')}" for c in notes.get("key_concepts", []) if isinstance(c, dict)]
        L += ["", "### Cheat Sheet"] + [f"- {x}" for x in notes.get("cheat_sheet", [])]
        L += ["", "### Common Mistakes"] + [f"- {x}" for x in notes.get("common_mistakes", [])] + [""]
    if quiz.get("questions"):
        L += ["## Quiz"]
        for i, q in enumerate(quiz["questions"], 1):
            L.append(f"**Q{i} ({q.get('difficulty', '')}).** {q.get('question')}")
            L += [f"   {chr(65 + j)}. {o}" for j, o in enumerate(q.get("options", []))]
            ai = q.get("answer_index", 0)
            ans = chr(65 + ai) if 0 <= ai < 26 else "?"
            L += [f"   *Answer: {ans} - {q.get('explanation', '')}*", ""]
    if rev:
        L += ["## Review", f"Score {rev.get('overall')} / 10 ({rev.get('verdict')})"] + [f"- {s}" for s in rev.get("suggestions", [])[:5]]
    L += ["", "*Resources are suggestions by a local LLM - search for them by name and verify before use.*"]
    return "\n".join(L)
