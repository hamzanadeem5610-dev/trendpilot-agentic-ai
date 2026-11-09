"""Testing + error analysis (brief sections 16 and 17).

* `TEST_CASES` - the 10 test inputs from the brief.
* `run_suite`  - runs every case through the agent and records: input, tools used, final output,
                 saved file, review score, rule issues, time, and whether the result was useful.
* The suite is run twice: **v1** (baseline prompts, fixed pipeline, no review loop) and **v2** (improved
  prompts + LLM planner + reviewer loop). `error_analysis` then takes the weak v1 responses, explains the
  problem and the likely reason, names the fix that v2 applied, and shows whether the fix worked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .quality import script_seconds

TEST_CASES = [
    {"no": 1, "topic": "YOLOv8 helmet detection", "platform": "LinkedIn", "tone": "Professional", "output_type": "Post + hashtags"},
    {"no": 2, "topic": "AI internship experience", "platform": "LinkedIn", "tone": "Motivational", "output_type": "Post"},
    {"no": 3, "topic": "Python learning journey", "platform": "Instagram Reels", "tone": "Funny", "output_type": "Reel script"},
    {"no": 4, "topic": "Machine learning project", "platform": "X / Twitter", "tone": "Short and catchy", "output_type": "Short post"},
    {"no": 5, "topic": "Data annotation struggles", "platform": "LinkedIn", "tone": "Humorous", "output_type": "Post"},
    {"no": 6, "topic": "Model failed in production", "platform": "Instagram Reels", "tone": "Meme style", "output_type": "Reel script"},
    {"no": 7, "topic": "Final year project", "platform": "LinkedIn", "tone": "Professional", "output_type": "Portfolio post"},
    {"no": 8, "topic": "Computer vision project", "platform": "YouTube Shorts", "tone": "Exciting", "output_type": "Video script"},
    {"no": 9, "topic": "Debugging FastAPI", "platform": "X / Twitter", "tone": "Funny", "output_type": "Short post"},
    {"no": 10, "topic": "Open-source contribution", "platform": "LinkedIn", "tone": "Inspirational", "output_type": "Post"},
]

# problem code -> (possible reason, fix applied in v2)
REASON_FIX = {
    "too_long": ("The baseline prompt gave no length limit for the platform", "Added per-platform word/character limits to the caption prompt; the reviewer loop re-runs the caption if it is still too long"),
    "too_short": ("No target length in the prompt", "Added a target word range per platform"),
    "over_char_limit": ("X/Twitter's 280-character limit (incl. hashtags) was never mentioned", "Added the 280-character rule and an automatic character-count check"),
    "generic_phrases": ("Prompt was not specific enough, so the model fell back on cliches", "Added target audience, concrete details from the topic and a banned-cliche list to the prompt"),
    "weak_hook": ("No guidance on what makes a strong hook", "Added few-shot examples of strong vs weak hooks and a 20-word hook limit"),
    "missing_hook": ("Model skipped the hook field", "Structured JSON output with required fields + retry"),
    "missing_caption": ("Model returned an empty caption", "Structured JSON output with required fields + retry"),
    "hashtags_in_caption": ("Model mixed hashtags into the caption", "Caption prompt says 'no hashtags' and the tool strips trailing hashtags"),
    "possible_invented_metric": ("Small models invent numbers to sound impressive", "Rule: never invent numbers - use placeholders like [your mAP score]"),
    "too_many_hashtags": ("No hashtag count was given for the platform", "Platform-specific hashtag counts + automatic cap in the hashtag tool"),
    "too_few_hashtags": ("No hashtag count was given for the platform", "Platform-specific hashtag counts"),
    "irrelevant_hashtags": ("Hashtag tool prompt was weak (no topic context, spam tags allowed)", "Improved hashtag prompt: topic + broad/niche mix, spam tags banned, relevance check"),
    "missing_hashtags": ("Model returned no hashtags", "Structured JSON output + retry"),
    "script_too_long": ("No time limit specified for the video", "Added a 15-60 second limit with timestamps per scene; reviewer re-runs the script tool if too long"),
    "script_too_short": ("No minimum duration given", "Added a target duration per platform"),
    "script_too_few_scenes": ("Script structure was not specified", "Required 4-6 scenes: hook, explanation, payoff, closing"),
    "voiceover_too_long": ("No limit on spoken words per scene", "Limited voiceover to 25 words per scene"),
    "missing_closing": ("Prompt did not ask for a closing line", "Closing line made a required field"),
    "missing_script": ("Model returned no scenes", "Structured JSON output + retry"),
    "title_too_long": ("No title length limit", "Title limited to 60 characters"),
    "thumbnail_too_long": ("No thumbnail text limit", "Thumbnail text limited to 5 words"),
    "missing_title": ("Model returned no title", "Structured JSON output + retry"),
    "tool_failed": ("Model returned broken/invalid JSON for a tool", "Ollama structured outputs (JSON schema) + automatic retry with the parse error"),
    "low_review_score": ("Content judged generic or off-tone by the reviewer", "Reviewer loop: flagged tools are re-run with the reviewer's suggestions (max 2 rounds)"),
}
MAJOR_ORDER = ["tool_failed", "over_char_limit", "script_too_long", "too_long", "irrelevant_hashtags", "generic_phrases",
               "weak_hook", "too_many_hashtags", "possible_invented_metric", "missing_hook", "missing_caption",
               "missing_hashtags", "missing_script", "title_too_long", "script_too_few_scenes", "too_short",
               "too_few_hashtags", "hashtags_in_caption", "voiceover_too_long", "thumbnail_too_long", "missing_closing",
               "script_too_short", "missing_title", "low_review_score"]


def _useful(rev: dict) -> str:
    score = rev.get("overall") or 0
    majors = [i for i in rev.get("rule_issues", []) if i["severity"] == "major"]
    if score >= 7.5 and not majors:
        return "Yes"
    if score >= 6:
        return "Partly"
    return "No"


def summarise(case: dict, res: dict, root: Path) -> dict:
    a, rev = res["artifacts"], res["review"]
    sc = a.get("reel_script") or {}
    fails = [e["name"] for e in res["trace"] if e["kind"] == "error"]
    codes = [i["code"] for i in rev.get("rule_issues", [])] + (["tool_failed"] if fails else [])
    if (rev.get("overall") or 0) < 7 and not codes:
        codes.append("low_review_score")
    saved = res["saved_files"].get("markdown", "")
    try:
        saved = str(Path(saved).relative_to(root))
    except ValueError:
        pass
    preview = a.get("hook") or (sc.get("scenes") or [{}])[0].get("voiceover", "")
    return {
        **case, "prompt_version": res["prompt_version"],
        "input_prompt": f"Topic: {case['topic']} | Platform: {case['platform']} | Tone: {case['tone']} | Output: {case['output_type']}",
        "tools_used": res["tools_used"], "review_score": rev.get("overall"), "llm_score": rev.get("llm_overall"),
        "verdict": rev.get("verdict"), "revisions": res["revisions"], "issue_codes": codes,
        "issues": [i["message"] for i in rev.get("rule_issues", [])] + [f"Tool failed: {f}" for f in fails],
        "useful": _useful(rev), "seconds": res.get("total_seconds"), "llm_calls": res.get("stats", {}).get("llm_calls"),
        "hook": a.get("hook", ""), "caption": a.get("caption", ""), "caption_words": len((a.get("caption") or "").split()),
        "hashtags": a.get("hashtags", []), "script_seconds": script_seconds(sc) if sc else None,
        "script_scenes": len(sc.get("scenes") or []), "title": (a.get("titles") or {}).get("best_title", ""),
        "preview": preview[:160], "saved_file": saved, "review_suggestions": rev.get("suggestions", [])[:3],
    }


def run_suite(agent, cases=TEST_CASES, label: str = "", out_dir: str | Path | None = None) -> list[dict]:
    """Run all test cases with `agent` (its settings decide v1/v2). Results are saved as JSON."""
    rows = []
    root = agent.settings.outputs_dir.parent
    for case in cases:
        t0 = time.time()
        try:
            res = agent.run(case["topic"], case["platform"], case["tone"], case["output_type"])
            rows.append(summarise(case, res, root))
        except Exception as e:                    # one broken case must not stop the suite
            rows.append({**case, "prompt_version": agent.settings.prompt_version, "error": str(e)[:200],
                         "issue_codes": ["tool_failed"], "issues": [str(e)[:200]], "useful": "No", "review_score": 0,
                         "tools_used": [], "seconds": round(time.time() - t0, 1), "saved_file": ""})
        print(f"  [{label or agent.settings.prompt_version}] test {case['no']:>2}: score {rows[-1].get('review_score')} | "
              f"useful {rows[-1]['useful']} | issues {rows[-1]['issue_codes']}", flush=True)
    out = Path(out_dir or agent.settings.logs_dir) / f"test_results_{label or agent.settings.prompt_version}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def compare(v1: list[dict], v2: list[dict]) -> list[dict]:
    by = {r["no"]: r for r in v1}
    out = []
    for r in v2:
        b = by.get(r["no"], {})
        out.append({"Test": r["no"], "Topic": r["topic"], "Platform": r["platform"],
                    "v1 score": b.get("review_score"), "v1 issues": len(b.get("issue_codes", [])), "v1 useful": b.get("useful"),
                    "v2 score": r.get("review_score"), "v2 issues": len(r.get("issue_codes", [])), "v2 useful": r.get("useful"),
                    "v2 self-revisions": r.get("revisions")})
    return out


def error_analysis(v1: list[dict], v2: list[dict], min_rows: int = 5, manual_notes: dict | None = None) -> list[dict]:
    """Weak/failed responses -> problem, possible reason, fix applied, result after the fix."""
    manual_notes = manual_notes or {}
    v2_by = {r["no"]: r for r in v2}
    rows = []
    for r in sorted(v1, key=lambda x: (x.get("review_score") or 0)):
        codes = sorted(r.get("issue_codes", []), key=lambda c: MAJOR_ORDER.index(c) if c in MAJOR_ORDER else 99)
        if not codes and (r.get("review_score") or 10) >= 7:
            continue
        code = codes[0] if codes else "low_review_score"
        reason, fix = REASON_FIX.get(code, ("Prompt not specific enough", "Improved prompt"))
        after = v2_by.get(r["no"], {})
        fixed = code not in after.get("issue_codes", [])
        msg = next((m for m in r.get("issues", []) if m), f"Reviewer score only {r.get('review_score')}/10")
        rows.append({"Test Case": f"#{r['no']} {r['topic']} ({r['platform']}, {r['tone']})",
                     "Problem Found": msg, "Possible Reason": manual_notes.get(r["no"], reason), "Fix Applied": fix,
                     "Result After Fix (v2)": f"{'fixed' if fixed else 'still present'} - score {r.get('review_score')} -> {after.get('review_score')}",
                     "code": code})
    # if v1 had fewer weak cases than required, also list weak v2 results (honest "still open" items)
    for r in sorted(v2, key=lambda x: (x.get("review_score") or 0)):
        if len(rows) >= max(min_rows, len(rows)) and len(rows) >= min_rows:
            break
        if r.get("issue_codes") and not any(x["Test Case"].startswith(f"#{r['no']} ") for x in rows):
            code = r["issue_codes"][0]
            reason, fix = REASON_FIX.get(code, ("Model limitation", "Further prompt work"))
            rows.append({"Test Case": f"#{r['no']} {r['topic']} ({r['platform']}, {r['tone']}) [v2]",
                         "Problem Found": (r.get("issues") or ["weak output"])[0], "Possible Reason": reason,
                         "Fix Applied": "Open - next step: " + fix, "Result After Fix (v2)": f"score {r.get('review_score')}", "code": code})
    return rows


IMPROVEMENTS = [
    "Use a larger local model (e.g. Gemma 3 12B or Qwen 2.5 7B) for funnier, more natural humour and meme-style content.",
    "Add a human-in-the-loop step in the UI: show 3 hooks and let the user pick before the caption is written.",
    "Store platform analytics (likes/comments) back into memory so the agent learns which hooks perform best.",
    "Add a fact-checking tool that compares claims in the caption with the user's README / project files (RAG).",
    "Add image generation for the cover image prompt (e.g. Stable Diffusion locally) and export the post as PDF/carousel.",
    "Schedule posts and track a content calendar in SQLite; deploy the FastAPI backend with Docker.",
]


def write_test_cases_md(path: str | Path, v1: list[dict], v2: list[dict], errors: list[dict], model: str) -> Path:
    def esc(x):
        return str(x).replace("|", "/").replace("\n", " ")
    L = ["# TrendPilot - Test Cases and Results", "",
         f"Model: **{model}** (local, Ollama) · 10 test inputs from the task brief · each run twice: "
         "**v1** = baseline prompts / fixed pipeline / no review loop, **v2** = improved prompts + planner + reviewer loop.", "",
         "## Final results (v2 - improved agent)", "",
         "| # | Input prompt | Tools used | Final output (preview) | Saved file | Score | Useful? |", "|---|---|---|---|---|---|---|"]
    for r in v2:
        tools = " -> ".join(t.replace(" Generator", "").replace(" Tool", "") for t in r.get("tools_used", []))
        L.append(f"| {r['no']} | {esc(r.get('input_prompt', r['topic']))} | {esc(tools)} | {esc(r.get('preview', ''))} | "
                 f"`{esc(r.get('saved_file', ''))}` | {r.get('review_score')} | {r.get('useful')} |")
    L += ["", "## v1 (baseline) vs v2 (improved)", "", "| # | Topic | Platform | v1 score | v1 issues | v2 score | v2 issues | v2 self-revisions |",
          "|---|---|---|---|---|---|---|---|"]
    for c in compare(v1, v2):
        L.append(f"| {c['Test']} | {esc(c['Topic'])} | {c['Platform']} | {c['v1 score']} | {c['v1 issues']} | {c['v2 score']} | {c['v2 issues']} | {c['v2 self-revisions']} |")
    L += ["", "## Error analysis (weak / failed responses)", "",
          "| Test Case | Problem Found | Possible Reason | Fix Applied | Result After Fix |", "|---|---|---|---|---|"]
    for e in errors:
        L.append(f"| {esc(e['Test Case'])} | {esc(e['Problem Found'])} | {esc(e['Possible Reason'])} | {esc(e['Fix Applied'])} | {esc(e['Result After Fix (v2)'])} |")
    L += ["", "## Suggested improvements", ""] + [f"{i}. {s}" for i, s in enumerate(IMPROVEMENTS, 1)]
    L += ["", "## Full outputs", "Every run is saved as Markdown in `outputs/generated_posts/` or `outputs/generated_scripts/` "
          "and as JSON in `outputs/saved_results/`. Raw test logs: `outputs/logs/test_results_v1.json`, `test_results_v2.json`."]
    p = Path(path)
    p.write_text("\n".join(L), encoding="utf-8")
    return p
