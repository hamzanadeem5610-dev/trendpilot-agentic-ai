"""Build the final deliverables from the real run results.

    python scripts/build_report.py

Reads   outputs/logs/test_results_v1.json, test_results_v2.json, outputs/logs/showcase.json, screenshots/*.png
Writes  report/final_report.pdf               (brief section 20, item 11)
        report/linkedin_portfolio_report.md   (brief section 22 - LinkedIn-ready report)
        screenshots/test_scores_v1_vs_v2.png
        README.md results block (between the RESULTS markers)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from fpdf import FPDF                                                # noqa: E402
from fpdf.fonts import FontFace                                      # noqa: E402

from app.evaluation import IMPROVEMENTS, compare, error_analysis     # noqa: E402
from app.tools import REGISTRY                                       # noqa: E402

AUTHOR = "Hamza Nadeem"
TITLE = "TrendPilot - A Local Agentic AI Content Planner (with StudyBuddy mode)"
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
OUT, SHOTS, LOGS = ROOT / "outputs", ROOT / "screenshots", ROOT / "outputs" / "logs"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def clean(t) -> str:
    """Strip emoji / characters the PDF font cannot draw."""
    t = str(t if t is not None else "")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", " - ")
    return "".join(ch for ch in t if ord(ch) < 0x2600 and not 0xFE00 <= ord(ch) <= 0xFE0F)


def load():
    v1 = json.loads((LOGS / "test_results_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((LOGS / "test_results_v2.json").read_text(encoding="utf-8"))
    show = json.loads((LOGS / "showcase.json").read_text(encoding="utf-8")) if (LOGS / "showcase.json").exists() else {}
    example = json.loads(Path(show["example_json"]).read_text(encoding="utf-8")) if show.get("example_json") and Path(show["example_json"]).exists() else None
    revised = json.loads(Path(show["revised_json"]).read_text(encoding="utf-8")) if show.get("revised_json") and Path(show["revised_json"]).exists() else None
    study = json.loads(Path(show["study_json"]).read_text(encoding="utf-8")) if show.get("study_json") and Path(show["study_json"]).exists() else None
    from app.agent import TrendPilotAgent
    for r in (example, revised):
        if r is not None and "final_output" not in r:
            r["final_output"] = TrendPilotAgent.final_output(r)
    return v1, v2, show, example, revised, study


def stats(v1, v2):
    avg = lambda rows: round(sum((r.get("review_score") or 0) for r in rows) / max(len(rows), 1), 2)
    iss = lambda rows: sum(len(r.get("issue_codes", [])) for r in rows)
    useful = lambda rows: sum(r.get("useful") == "Yes" for r in rows)
    saved = len(list((OUT / "generated_posts").glob("*.md"))) + len(list((OUT / "generated_scripts").glob("*.md")))
    return {"v1_avg": avg(v1), "v2_avg": avg(v2), "v1_issues": iss(v1), "v2_issues": iss(v2), "v1_useful": useful(v1),
            "v2_useful": useful(v2), "n": len(v2), "saved_md": saved, "saved_json": len(list((OUT / "saved_results").glob("*.json"))),
            "revised_cases": sum(1 for r in v2 if r.get("revisions")),
            "avg_seconds": round(sum((r.get("seconds") or 0) for r in v2) / max(len(v2), 1), 1),
            "strong": [r for r in v2 if r.get("useful") == "Yes"], "weak": [r for r in v2 if r.get("useful") != "Yes"]}


def score_chart(v1, v2, path: Path) -> Path:
    labels = [f"#{r['no']}" for r in v2]
    s1 = [(next((x for x in v1 if x["no"] == r["no"]), {}).get("review_score") or 0) for r in v2]
    s2 = [(r.get("review_score") or 0) for r in v2]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], s1, w, color=BLUE, label="v1 baseline")
    ax.bar([i + w / 2 + 0.01 for i in x], s2, w, color=ORANGE, label="v2 improved")
    ax.axhline(7.5, color=MUTED, lw=1, ls="--")
    ax.text(len(labels) - 0.5, 7.65, "ready threshold 7.5", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(list(x), labels, fontsize=9, color=INK)
    ax.set_ylim(0, 10.5); ax.set_ylabel("Review score (0-10)", fontsize=9, color=MUTED)
    ax.set_title("Review score per test case: v1 baseline vs v2 improved agent", fontsize=11, loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#c9c8c3"); ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ecebe8", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white"); plt.close(fig)
    return path


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("DV", "", 8); self.set_text_color(120)
            self.cell(0, 6, f"{AUTHOR} - AIRI Team PITB AI Internship Task 2 - TrendPilot", align="R")
            self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("DV", "", 8); self.set_text_color(120)
        self.cell(0, 6, f"Page {self.page_no()}", align="C"); self.set_text_color(0)


def build_pdf(v1, v2, show, example, revised, study, st, errors, path: Path) -> Path:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(True, 15)
    pdf.set_margins(16, 14, 16)
    pdf.add_font("DV", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DV", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DV", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.add_font("MONO", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
    W = pdf.epw

    def keep(space=45):                                       # don't leave a heading alone at the bottom of a page
        if pdf.get_y() > pdf.h - space:
            pdf.add_page()

    def h1(t):
        keep(); pdf.ln(2); pdf.set_font("DV", "B", 15); pdf.set_text_color(26, 74, 140)
        pdf.multi_cell(W, 8, clean(t)); pdf.set_text_color(0); pdf.ln(1)

    def h2(t):
        keep(40); pdf.ln(1); pdf.set_font("DV", "B", 11.5); pdf.multi_cell(W, 6.5, clean(t)); pdf.ln(0.5)

    def para(t, size=10, style=""):
        pdf.set_font("DV", style, size); pdf.multi_cell(W, 5.4, clean(t)); pdf.ln(1.5)

    def bullets(items, size=10):
        pdf.set_font("DV", "", size)
        for it in items:
            pdf.set_x(pdf.l_margin + 3); pdf.multi_cell(W - 3, 5.3, "- " + clean(it)); pdf.ln(0.4)
        pdf.ln(1)

    def mono(t, size=8):
        pdf.set_font("MONO", "", size); pdf.set_fill_color(244, 244, 242)
        pdf.multi_cell(W, 4.2, clean(t), fill=True); pdf.ln(2)

    def image(p, caption, w=None):
        p = Path(p)
        if not p.exists():
            para(f"[screenshot missing: {p.name}]", 9, "I"); return
        w = w or W
        from PIL import Image
        iw, ih = Image.open(p).size
        h = w * ih / iw
        if h > 235:                                           # keep tall screenshots on one page
            h = 235; w = h * iw / ih
        avail = pdf.h - 15 - pdf.get_y() - 10
        if h > avail:
            if avail >= 0.6 * h:                              # shrink a little instead of leaving a blank half page
                w, h = w * avail / h, avail
            else:
                pdf.add_page()
        pdf.image(str(p), x=pdf.l_margin + (W - w) / 2, w=w, h=h)
        pdf.set_font("DV", "I", 8.5); pdf.set_text_color(90)
        pdf.multi_cell(W, 5, clean(caption), align="C"); pdf.set_text_color(0); pdf.ln(2)

    def table(header, rows, widths, size=7.6):
        pdf.set_font("DV", "", size)
        with pdf.table(col_widths=widths, text_align="LEFT", line_height=size * 0.5,
                       headings_style=FontFace(emphasis="BOLD", fill_color=(232, 240, 254)), padding=1) as t:
            r = t.row()
            for hcell in header:
                r.cell(clean(hcell))
            for row in rows:
                r = t.row()
                for c in row:
                    r.cell(clean(c))
        pdf.ln(3)

    model = show.get("model", v2[0].get("model", "gemma3:4b") if v2 else "gemma3:4b")
    # ---------------------------------------------------------------- cover
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("DV", "B", 24); pdf.set_text_color(26, 74, 140)
    pdf.multi_cell(W, 12, "TrendPilot", align="C"); pdf.set_text_color(0)
    pdf.set_font("DV", "", 14)
    pdf.multi_cell(W, 8, "A Local Agentic AI Content Planner\nwith a StudyBuddy mode", align="C"); pdf.ln(6)
    pdf.set_font("DV", "", 11)
    pdf.multi_cell(W, 7, f"AIRI Team PITB - AI Internship Task 2\nEnd-to-End Agentic AI Assistant using a Local LLM (Ollama)", align="C")
    pdf.ln(14)
    pdf.set_font("DV", "B", 12); pdf.multi_cell(W, 7, f"Submitted by: {AUTHOR}", align="C")
    pdf.set_font("DV", "", 11)
    pdf.multi_cell(W, 7, f"Model: {model} (local, via Ollama {show.get('ollama_version', '')})\n"
                         f"Hardware: {show.get('gpu', 'Google Colab GPU')}\nDate: {date.today():%d %B %Y}", align="C")
    pdf.ln(16)
    pdf.set_font("DV", "", 10)
    kpis = [("10", "test cases"), (f"{st['v1_avg']} -> {st['v2_avg']}", "avg review score v1 -> v2"),
            (f"{st['v1_issues']} -> {st['v2_issues']}", "rule violations v1 -> v2"), (str(st["saved_md"]), "saved outputs (.md)")]
    cw = W / 4
    pdf.set_font("DV", "B", 14)
    for k, _ in kpis:
        pdf.cell(cw, 9, clean(k), align="C")
    pdf.ln(9); pdf.set_font("DV", "", 8.5); pdf.set_text_color(90)
    for _, lbl in kpis:
        pdf.cell(cw, 5, lbl, align="C")
    pdf.set_text_color(0)

    # ---------------------------------------------------------------- 1 problem
    pdf.add_page()
    h1("1. Problem Understanding and Project Idea")
    para("Students and junior engineers build good projects but struggle to present them: posts are generic, hooks are weak, "
         "hashtags are random and reel scripts run too long. At the same time, learners need structured study plans instead "
         "of random tutorials. The task asked for an agent - not a chatbot - that understands the request, plans, calls tools, "
         "reviews its own work and saves the result, running fully locally through Ollama.")
    para("From the suggested ideas I combined Option A (TrendPilot - content planner) and Option B (StudyBuddy). TrendPilot is "
         "the main agent: it turns any topic into a platform-ready content plan (idea, audience, hook, caption, hashtags, reel "
         "script, title, image prompt). StudyBuddy reuses the same agent engine with different tools (study planner, notes, quiz, "
         "progress tracking). One engine with two tool sets shows the main idea of agentic AI: the planner decides which tools a "
         "request needs.")
    h2("Chatbot vs agent - what this project adds")
    table(["Simple chatbot", "TrendPilot agent"],
          [["One prompt -> one answer", "Understands the request, makes a plan and validates it"],
           ["No tools", f"{len([t for t in REGISTRY.values() if t.mode != 'study'])} content tools + 5 StudyBuddy tools, chosen by an LLM planner"],
           ["No quality control", "Content Reviewer (LLM rubric + deterministic platform rules) and an automatic rewrite loop"],
           ["Forgets everything", "SQLite memory: past runs, learned preferences, study progress"],
           ["Output only on screen", "Markdown + JSON files saved automatically; CLI, Streamlit and FastAPI demos"]], (50, 128), 8.5)

    # ---------------------------------------------------------------- 2 setup
    h1("2. Ollama Setup and Local Model")
    para(f"The model runs locally with Ollama on a Google Colab GPU, so no paid API is used. Installation: "
         f"`curl -fsSL https://ollama.com/install.sh | sh`, `ollama serve` in the background, then `ollama pull {model}`. "
         f"Python talks to the REST API (http://localhost:11434) with `requests`. Small fallback models (llama3.2:3b, qwen2.5:3b, "
         f"phi3:mini) are used automatically if the main model is not available.")
    bullets(["Structured outputs: every tool sends a JSON schema in Ollama's `format` field, so a 4B model returns valid JSON; "
             "if parsing still fails the error is sent back to the model and it retries (max 2).",
             "temperature 0.7 for creative tools, 0.2-0.5 for reviewer/planner; num_ctx 4096; keep_alive keeps the model in GPU memory.",
             "The test prompt from the brief ('Give me 5 viral LinkedIn post ideas about AI internships.') was run - see the screenshot."])
    image(SHOTS / "ollama_running.png", "Screenshot 1 - Ollama running the local model (ollama --version, ollama list, test prompt).")

    # ---------------------------------------------------------------- 3 workflow
    pdf.add_page()
    h1("3. Agent Workflow Design")
    image(SHOTS / "architecture.png", "Figure - TrendPilot agent graph (LangGraph StateGraph).")
    bullets(["Step 1 - User input: topic, platform, tone, output type.",
             "Step 2 - Memory + research: recall similar past runs and learned preferences; optional DuckDuckGo trend search.",
             "Step 3 - Understand: the LLM extracts goal, target audience and key points as JSON.",
             "Step 4 - Plan: the LLM chooses tools from a registry; the plan is validated (unknown tools removed, missing required tools added, correct order).",
             "Step 5 - Execute: each tool runs and prints [Tool Called]: <name>.",
             "Step 6 - Review: Content Reviewer scores hook, clarity, platform fit, tone match and engagement (0-10) and runs platform rules "
             "(length, character limit, hashtag count, script duration, cliches, invented numbers).",
             "Step 7 - Refine loop: if the score is below 7.5 or a major rule is broken, only the flagged tools are re-run with the "
             "reviewer's suggestions (max 2 rounds).",
             "Step 8 - Save + final output: File Saver writes Markdown + JSON, memory stores the run, the final output is returned.",
             "User feedback loop: 'make the hook shorter' -> an LLM router picks the tools to re-run, and a _revised file is saved."], 9.5)

    # ---------------------------------------------------------------- 4 tools
    h1("4. Tools Implemented")
    rows = [[t.display_name, {"content": "TrendPilot", "study": "StudyBuddy", "shared": "Both"}[t.mode],
             "Yes" if t.uses_llm else "No", t.description] for t in REGISTRY.values()]
    table(["Tool", "Mode", "LLM", "What it does"], rows, (38, 20, 10, 110), 7.8)

    # ---------------------------------------------------------------- 5 memory
    h1("5. Memory and File Saving")
    bullets(["Short-term memory: the LangGraph state carries the request, plan, artifacts, review history and trace during a run.",
             "Long-term memory (SQLite, outputs/memory.db): every run with topic, platform, tone, hook and score; learned preferences "
             "(favourite platform/tone); study plans and day-by-day progress with quiz scores.",
             "Memory is used, not just stored: similar past topics are recalled (keyword similarity) and injected into the prompt so the "
             "agent avoids repeating the same hook.",
             "File saving: posts -> outputs/generated_posts/*.md, scripts -> outputs/generated_scripts/*.md, full results -> "
             f"outputs/saved_results/*.json, study packs -> outputs/study_plans/. This run produced {st['saved_md']} Markdown and "
             f"{st['saved_json']} JSON files.",
             "Every run is also logged to outputs/logs/agent_runs.jsonl (tools, score, time, LLM calls) for evaluation."], 9.5)
    image(SHOTS / "saved_file_output.png", "Screenshot 3 - saved output files and the content of a saved post.")

    # ---------------------------------------------------------------- 6 demo
    pdf.add_page()
    h1("6. Demo and Outputs")
    para("Three demos are included: CLI (python demo.py - interactive, with a feedback loop), Streamlit (app/streamlit_app.py - "
         "topic, platform, tone, output type, Generate, final output, Save result, live agent trace, review chart, StudyBuddy tab) and "
         "FastAPI (app/main.py - POST /generate-content, /improve-content, /study-plan, /study-progress, GET /history).")
    image(SHOTS / "demo_output.png", "Screenshot 2 - CLI demo: [Tool Called] logs and the final output.")
    if (SHOTS / "streamlit_demo.png").exists():
        image(SHOTS / "streamlit_demo.png", "Screenshot - Streamlit demo after one generation.", w=W * 0.85)
    if example:
        fo = example["final_output"]
        h2("Example output (main example from the brief)")
        para(f"Input: {fo['topic']} | {fo['platform']} | {fo['tone']} | {example['request'].get('output_type')}", 9, "I")
        para(f"Tools used: {' -> '.join(example['tools_used'])}", 8.5)
        table(["Field", "Agent output"],
              [["Content idea", fo["content_idea"]], ["Target audience", fo["target_audience"]], ["Hook", fo["hook"]],
               ["Caption", fo["caption"]], ["Hashtags", " ".join(fo["hashtags"])],
               ["Reel script", fo["reel_script"]], ["Title / thumbnail", f"{fo['title']} / {fo['thumbnail_text']}"],
               ["Review", f"{fo['review_score']}/10 - " + "; ".join(map(str, fo["review_comments"][:3]))],
               ["Saved file", fo["saved_file"]]], (32, 146), 8)
    if revised:
        h2("User feedback loop")
        para(f"Feedback: \"{revised.get('user_feedback', show.get('feedback', ''))}\" -> tools re-run: "
             f"{', '.join(revised.get('revised_tools', []) or ['see trace'])}. New hook: {revised['artifacts'].get('hook', '')}", 9)
    if study:
        h2("StudyBuddy mode")
        plan = study["artifacts"].get("study_plan", {})
        para(f"Request: {study['request']['topic']} - {study['request'].get('days')} days, {study['request'].get('level')}. "
             f"Output: '{plan.get('title', '')}' with {len(plan.get('days', []))} daily plans, "
             f"{len(study['artifacts'].get('notes', {}).get('key_concepts', []))} key concepts and "
             f"{len(study['artifacts'].get('quiz', {}).get('questions', []))} quiz questions; progress is tracked in SQLite.", 9)
        table(["Day", "Focus", "Tasks"], [[d.get("day"), d.get("focus", ""), "; ".join(d.get("tasks", [])[:3])] for d in plan.get("days", [])[:7]],
              (12, 50, 116), 7.8)

    # ---------------------------------------------------------------- 7 testing
    pdf.add_page()
    h1("7. Testing (10 test cases)")
    para(f"All 10 inputs from the brief were run twice: v1 = baseline (simple prompts, fixed pipeline, no review loop) and "
         f"v2 = final agent (improved prompts, LLM planner, reviewer loop). Average review score went from {st['v1_avg']} to "
         f"{st['v2_avg']}, rule violations from {st['v1_issues']} to {st['v2_issues']}, and useful outputs from {st['v1_useful']}/10 "
         f"to {st['v2_useful']}/10. In v2 the reviewer loop automatically rewrote {st['revised_cases']} of 10 cases. "
         f"Average time per case: {st['avg_seconds']} s.")
    image(SHOTS / "test_scores_v1_vs_v2.png", "Figure - review score per test case (dashed line = ready threshold).")
    table(["#", "Input", "Tools used", "Final output (preview)", "Score", "Useful"],
          [[r["no"], f"{r['topic']} | {r['platform']} | {r['tone']} | {r['output_type']}",
            ", ".join(t.replace(" Generator", "").replace(" Tool", "") for t in r.get("tools_used", []) if "Memory" not in t),
            r.get("preview", ""), r.get("review_score"), r.get("useful")] for r in v2], (7, 40, 46, 61, 12, 12), 7)
    h2("v1 baseline vs v2 improved")
    table(["#", "Topic", "Platform", "v1 score", "v1 issues", "v2 score", "v2 issues", "Self-revisions"],
          [[c["Test"], c["Topic"], c["Platform"], c["v1 score"], c["v1 issues"], c["v2 score"], c["v2 issues"], c["v2 self-revisions"]]
           for c in compare(v1, v2)], (8, 46, 30, 17, 17, 17, 17, 26), 7.8)
    if st["strong"]:
        para("Strong outputs: " + "; ".join(f"#{r['no']} {r['topic']} ({r['review_score']})" for r in st["strong"][:6]) + ".", 9)
    if st["weak"]:
        para("Weaker outputs: " + "; ".join(f"#{r['no']} {r['topic']} ({r['review_score']}) - {(r.get('issues') or ['low reviewer score'])[0]}"
                                           for r in st["weak"]) + ".", 9)

    # ---------------------------------------------------------------- 8 error analysis
    h1("8. Error Analysis")
    table(["Test case", "Problem found", "Possible reason", "Fix applied", "Result after fix"],
          [[e["Test Case"], e["Problem Found"], e["Possible Reason"], e["Fix Applied"], e["Result After Fix (v2)"]] for e in errors],
          (30, 38, 38, 46, 26), 7)
    notes = show.get("manual_notes") or {}
    if notes:
        h2("Manual review notes")
        bullets([f"Test #{k}: {v}" for k, v in notes.items()], 9)

    # ---------------------------------------------------------------- 9 learnings
    h1("9. What I Learned")
    bullets(["How to run and serve a local LLM with Ollama on a Colab GPU and call it from Python through its REST API.",
             "The difference between a chatbot and an agent: planning, tool selection, state, self-review and memory.",
             "How to force a small model to return reliable JSON (schemas, validation and retry with the error message).",
             "That small models need specific prompts: audience, platform rules, few-shot examples and length limits changed the results a lot.",
             "How to combine an LLM judge with deterministic rule checks so the agent can find and fix its own mistakes.",
             "How to build LangGraph state machines with conditional edges (review -> refine loop).",
             "How to test an agent systematically with fixed test cases, baseline vs improved versions and an error-analysis table."], 9.5)
    h1("10. Future Improvements")
    bullets(IMPROVEMENTS, 9.5)
    h1("11. Repository Structure")
    mono("agentic_ai_task_2/\n  app/ main.py (FastAPI) agent.py (LangGraph) tools.py memory.py prompts.py llm.py quality.py\n"
         "       evaluation.py config.py streamlit_app.py\n  outputs/ generated_posts/ generated_scripts/ saved_results/ study_plans/ logs/\n"
         "  tests/ test_cases.md   screenshots/   report/ final_report.pdf linkedin_portfolio_report.md\n"
         "  notebooks/ TrendPilot_Colab.ipynb   scripts/   README.md   requirements.txt   demo.py", 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def linkedin_md(v1, v2, show, example, st) -> str:
    model = show.get("model", "gemma3:4b")
    tools = [t for t in REGISTRY.values() if t.mode in ("content", "shared")]
    ex = example["final_output"] if example else None
    L = ["# TrendPilot - A Local Agentic AI Content Planner Using Ollama and Gemma 3:4B", f"*{AUTHOR} - AIRI Team PITB AI Internship Task 2*", "",
         "## 1. Project Overview",
         f"I built TrendPilot, a local Agentic AI assistant that runs on {model} through Ollama, so no paid API is needed. It turns any "
         "project or topic into a platform-ready content plan: a content idea, target audience, hook, caption, hashtags, a short-video "
         "script, title and thumbnail text. Instead of answering in one shot, the agent plans which tools it needs, calls them, reviews "
         "its own output against platform rules and rewrites weak parts before saving the result. The same engine also runs a StudyBuddy "
         "mode that creates study plans, notes and quizzes and tracks progress.", "",
         "## 2. Tools and Technologies Used",
         f"Python, Ollama, {model} (fallbacks: Llama 3.2 3B, Qwen 2.5 3B, Phi-3 Mini), LangGraph, Streamlit, FastAPI, SQLite memory, "
         "JSON/Markdown file saving, DuckDuckGo search (optional), Google Colab (T4 GPU) and GitHub.", "",
         "## 3. Agent Workflow",
         "User gives topic -> memory recall -> agent understands the goal and audience -> planner selects tools -> tools generate content "
         "-> Content Reviewer scores it and checks platform rules -> weak parts are rewritten automatically -> File Saver stores Markdown + "
         "JSON -> final output is returned. The user can then give feedback (for example \"make the hook shorter\") and only the affected "
         "tools are re-run.", "", "## 4. Tools Implemented"]
    L += [f"- **{t.display_name}** - {t.description}" for t in tools]
    L += ["- **StudyBuddy tools** - Study Planner, Notes Generator, Quiz Generator, Study Pack Reviewer and Progress Saver.", "",
          "## 5. Demo and Outputs",
          "The project has three demos: an interactive CLI (`python demo.py`), a Streamlit app (topic, platform, tone, output type, "
          "Generate, final output, Save result, live agent trace) and a FastAPI backend (`POST /generate-content`)."]
    if ex:
        L += ["", f"Example input: *{ex['topic']}* - {ex['platform']} - {ex['tone']}", "", f"Example hook: \"{ex['hook']}\"", "",
              f"Hashtags: {' '.join(ex['hashtags'])}", "", f"Saved file: `{ex['saved_file']}`"]
    L += ["", "Screenshots: `screenshots/ollama_running.png`, `screenshots/demo_output.png`, `screenshots/saved_file_output.png`.", "",
          "## 6. Testing",
          f"I tested the agent on {st['n']} inputs from the brief (YOLOv8 posts, internship stories, funny Python reels, X posts, meme-style "
          f"reels, portfolio posts and YouTube Shorts). The improved agent reached an average review score of {st['v2_avg']}/10 "
          f"(baseline {st['v1_avg']}) and {st['v2_useful']}/10 outputs were directly usable (baseline {st['v1_useful']}/10)."
          + (f" Weaker cases: {', '.join(r['topic'] for r in st['weak'])}." if st["weak"] else ""), "",
          "## 7. Error Analysis",
          f"The first version broke platform rules {st['v1_issues']} times: captions too long for X, too many and irrelevant hashtags, "
          "reel scripts longer than 60 seconds and generic cliches. I added platform-specific limits, target audience, few-shot hook "
          "examples and a banned-cliche list to the prompts, plus a reviewer loop that re-runs only the weak tool. Violations dropped to "
          f"{st['v2_issues']}.", "",
          "## 8. What I Learned",
          "- How to run local LLMs with Ollama and call them from Python.", "- The difference between a chatbot and an AI agent.",
          "- How to connect tools to an LLM with a planner and a tool registry.", "- Prompt design for structured JSON outputs with small models.",
          "- How to save agent results and use SQLite memory.", "- How to make an agent review and improve its own work.",
          "- How to test, compare versions and analyse failures of an agentic workflow.", "",
          "## 9. Future Improvements"] + [f"- {s}" for s in IMPROVEMENTS] + ["",
          "## 10. LinkedIn Post", "",
          "Today, I completed AIRI Team PITB - AI Internship Task 2, where I built an end-to-end Agentic AI assistant using a local LLM through Ollama.", "",
          f"For this task, I developed TrendPilot, an AI agent that takes a topic and generates content ideas, LinkedIn posts, hashtags, reel "
          f"scripts and saved Markdown files - running fully locally on {model}. The agent plans which tools to use, reviews its own output "
          f"against platform rules and rewrites weak parts automatically. On 10 test cases, the improved version raised the average review "
          f"score from {st['v1_avg']} to {st['v2_avg']}/10. It also has a StudyBuddy mode for study plans, notes and quizzes.", "",
          "This project helped me understand how agentic AI systems work beyond simple chatbot responses. I learned how to design an agent "
          "workflow, connect tools, add memory, save outputs, test different prompts, and analyze weak responses.", "",
          f"Tools used: Python, Ollama, {model}, LangGraph, Streamlit, FastAPI, SQLite, JSON/Markdown, and GitHub.", "",
          "Key learning: Agentic AI is not only about generating text. It is about planning, using tools, storing useful information, "
          "improving outputs, and completing tasks in a structured way.", "",
          "#AgenticAI #Ollama #LocalLLM #Gemma #Python #ArtificialIntelligence #AIRI #PITB #AIInternship #MachineLearning #GenerativeAI"]
    return "\n".join(L)


def update_readme(st, show):
    p = ROOT / "README.md"
    if not p.exists():
        return
    block = ("<!-- RESULTS:START -->\n"
             f"| Metric | v1 baseline | v2 final agent |\n|---|---|---|\n"
             f"| Average review score (0-10) | {st['v1_avg']} | **{st['v2_avg']}** |\n"
             f"| Platform-rule violations (10 tests) | {st['v1_issues']} | **{st['v2_issues']}** |\n"
             f"| Directly usable outputs | {st['v1_useful']}/10 | **{st['v2_useful']}/10** |\n\n"
             f"Model: `{show.get('model', 'gemma3:4b')}` on {show.get('gpu', 'Google Colab')} - avg {st['avg_seconds']} s per request - "
             f"{st['saved_md']} saved Markdown outputs. Full table: [tests/test_cases.md](tests/test_cases.md).\n\n"
             "![v1 vs v2](screenshots/test_scores_v1_vs_v2.png)\n"
             "<!-- RESULTS:END -->")
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, s, flags=re.S)
    p.write_text(s, encoding="utf-8")


def main():
    v1, v2, show, example, revised, study = load()
    st = stats(v1, v2)
    errors = error_analysis(v1, v2, manual_notes={int(k): v for k, v in (show.get("manual_notes") or {}).items()})
    score_chart(v1, v2, SHOTS / "test_scores_v1_vs_v2.png")
    pdf = build_pdf(v1, v2, show, example, revised, study, st, errors, ROOT / "report" / "final_report.pdf")
    (ROOT / "report" / "linkedin_portfolio_report.md").write_text(linkedin_md(v1, v2, show, example, st), encoding="utf-8")
    update_readme(st, show)
    print("Report:", pdf)
    print("LinkedIn report:", ROOT / "report" / "linkedin_portfolio_report.md")
    print({k: v for k, v in st.items() if k not in ("strong", "weak")})


if __name__ == "__main__":
    main()
