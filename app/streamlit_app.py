"""TrendPilot - Streamlit demo (brief section 15, Option B).

Run:  streamlit run app/streamlit_app.py
Fields required by the brief: Topic, Platform, Tone, Output type, Generate button, Final output display,
Save result button. Extras: live agent trace, review scores, feedback loop, memory/history, StudyBuddy mode.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent import TrendPilotAgent                                   # noqa: E402
from app.config import OUTPUT_TYPES, PLATFORMS, TONES, Settings          # noqa: E402
from app.llm import OllamaError                                         # noqa: E402
from app.tools import render_markdown, render_study_markdown            # noqa: E402

st.set_page_config(page_title="TrendPilot - Local Agentic AI", page_icon="🚀", layout="wide")


@st.cache_resource
def get_agent() -> TrendPilotAgent:
    s = Settings()
    s.verbose = False
    return TrendPilotAgent(s)


agent = get_agent()
qp = st.query_params

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🚀 TrendPilot")
    st.caption("Local Agentic AI · Ollama")
    try:
        h = agent.llm.health()
        st.success(f"Ollama {h['version']} · model **{agent.llm.model}** {'✅' if h['model_ready'] else '⚠️ not pulled'}")
    except OllamaError as e:
        st.error(str(e))
    mode = st.radio("Agent", ["TrendPilot (content)", "StudyBuddy (study plan)"], index=1 if qp.get("mode") == "study" else 0)
    st.divider()
    st.subheader("🧠 Memory")
    prefs = agent.memory.prefs()
    if prefs:
        st.caption(f"Favourite platform: {prefs.get('favourite_platform')} · tone: {prefs.get('favourite_tone')}")
    for r in agent.memory.history(6):
        st.caption(f"#{r['id']} {r['topic'][:38]} · {r['platform']} · {r['review_score']}")


def run_with_trace(fn, *args):
    lines = []
    with st.status("Agent working...", expanded=True) as status:
        box = st.empty()

        def on_event(ev):
            label = {"tool": "🛠️ Tool called", "step": "🧭 Step", "plan": "🗺️ Plan", "revise": "🔁 Revision",
                     "error": "❌ Failed", "skip": "⏭️ Skipped"}.get(ev["kind"], ev["kind"])
            lines.append(f"**{label}:** {ev['name']}" + (f" — `{str(ev['detail'])[:110]}`" if ev.get("detail") else ""))
            box.markdown("\n\n".join(lines))
        agent.on_event = on_event
        t0 = time.time()
        try:
            res = fn(*args)
        finally:
            agent.on_event = None
        status.update(label=f"Done in {time.time() - t0:.0f}s · {len(res['tools_used'])} tool calls", state="complete", expanded=False)
    return res


# ------------------------------------------------------------------ TrendPilot
if mode.startswith("TrendPilot"):
    st.header("Turn any topic into a viral content plan")
    c1, c2 = st.columns([2, 1])
    topic = c1.text_input("Topic", qp.get("topic", "I want to create a viral LinkedIn post about my YOLOv8 helmet detection project"))
    platform = c2.selectbox("Platform", PLATFORMS, index=PLATFORMS.index(qp.get("platform", "LinkedIn")) if qp.get("platform", "LinkedIn") in PLATFORMS else 0)
    c3, c4 = st.columns(2)
    tone = c3.selectbox("Tone", TONES, index=1)
    output_type = c4.selectbox("Output type", OUTPUT_TYPES, index=OUTPUT_TYPES.index("Post + Reel Script"))
    go = st.button("⚡ Generate", type="primary", width="stretch")
    if go or (qp.get("demo") == "1" and "result" not in st.session_state):
        try:
            st.session_state["result"] = run_with_trace(agent.run, topic, platform, tone, output_type)
        except OllamaError as e:
            st.error(str(e))

    res = st.session_state.get("result")
    if res and res.get("mode") == "content":
        a, rev, fo = res["artifacts"], res["review"], res["final_output"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Review score", f"{rev.get('overall')}/10")
        m2.metric("Verdict", rev.get("verdict", "-"))
        m3.metric("Self-revisions", res["revisions"])
        m4.metric("Time", f"{res.get('total_seconds', 0):.0f}s")
        t1, t2, t3, t4 = st.tabs(["📝 Final output", "🎬 Reel script", "🔍 Review", "🤖 Agent trace"])
        with t1:
            st.markdown(f"**Content idea:** {fo['content_idea']}  \n**Target audience:** {fo['target_audience']}")
            if fo["hook"]:
                st.info(f"**Hook:** {fo['hook']}")
            if fo["caption"]:
                st.text_area("Caption", fo["caption"], height=260)
            if fo["hashtags"]:
                st.code(" ".join(fo["hashtags"]), language=None)
            if fo["title"]:
                st.markdown(f"**Title:** {fo['title']}  \n**Thumbnail text:** {fo['thumbnail_text']}")
            st.caption(f"Auto-saved: `{fo['saved_file']}`")
        with t2:
            sc = a.get("reel_script") or {}
            if sc.get("scenes"):
                st.markdown(f"**~{sc.get('total_seconds', 0):.0f} seconds**")
                st.table([{k: s.get(k, "") for k in ("time", "visual", "voiceover", "on_screen_text")} for s in sc["scenes"]])
                st.markdown(f"**Closing line:** {sc.get('closing_line', '')}")
            else:
                st.caption("No reel script requested.")
        with t3:
            if rev.get("scores"):
                st.bar_chart({k.replace("_", " "): v for k, v in rev["scores"].items()})
            for s_ in rev.get("suggestions", []):
                st.markdown(f"- 💡 {s_}")
            for i in rev.get("issues", []):
                st.markdown(f"- ⚠️ {i}")
            for h in a.get("review_history", []):
                st.caption(f"Earlier round: score {h['overall']} → re-ran {h['rewrite']}")
        with t4:
            st.markdown("**Plan:** " + " → ".join(p["tool"] for p in res["plan"]))
            for f in res.get("plan_fixes", []):
                st.caption(f"plan check: {f}")
            st.dataframe([{k: e[k] for k in ("t", "kind", "name", "seconds")} | {"detail": str(e["detail"])[:120]} for e in res["trace"]],
                         width="stretch", hide_index=True)
            st.json(res.get("stats", {}))

        st.divider()
        cA, cB = st.columns([3, 1])
        fb = cA.text_input("Want changes? (feedback loop)", placeholder="e.g. make the hook shorter and funnier")
        if cB.button("🔁 Improve", width="stretch") and fb:
            st.session_state["result"] = run_with_trace(agent.revise, res, fb)
            st.rerun()
        md = render_markdown(res)
        cS, cD = st.columns(2)
        if cS.button("💾 Save result", width="stretch"):
            p = agent.settings.results_dir / (Path(fo["saved_file"]).stem + "_saved.md")
            p.write_text(md, encoding="utf-8")
            st.success(f"Saved to `{p}`")
        cD.download_button("⬇️ Download Markdown", md, file_name=Path(fo["saved_file"]).name or "trendpilot.md", width="stretch")

# ------------------------------------------------------------------ StudyBuddy
else:
    st.header("📚 StudyBuddy - your personal study plan agent")
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    topic = c1.text_input("What do you want to learn?", qp.get("topic", "Object detection with YOLOv8"))
    days = c2.number_input("Days", 1, 30, 7)
    level = c3.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    hours = c4.number_input("Hours/day", 0.5, 8.0, 2.0, 0.5)
    if st.button("📅 Create study pack", type="primary", width="stretch") or (qp.get("demo") == "1" and "study" not in st.session_state):
        try:
            st.session_state["study"] = run_with_trace(agent.run_study, topic, days, level, hours)
        except OllamaError as e:
            st.error(str(e))
    res = st.session_state.get("study")
    if res:
        a = res["artifacts"]
        pid = a.get("plan_id")
        prog = agent.memory.progress(pid) if pid else {}
        st.progress(prog.get("percent", 0) / 100, text=f"Progress: {prog.get('days_completed', 0)}/{prog.get('days_total', 0)} days")
        t1, t2, t3 = st.tabs(["📅 Plan", "🗒️ Notes", "❓ Quiz"])
        with t1:
            for d in a.get("study_plan", {}).get("days", []):
                done = any(r["day"] == d["day"] and r["completed"] for r in prog.get("days", []))
                with st.expander(f"{'✅' if done else '⬜'} Day {d['day']}: {d.get('focus', '')} ({d.get('minutes', 0):.0f} min)"):
                    st.markdown("\n".join([f"- 🎯 {g}" for g in d.get("goals", [])] + [f"- {t}" for t in d.get("tasks", [])]))
                    if not done and st.button(f"Mark day {d['day']} done", key=f"done{d['day']}"):
                        agent.memory.update_progress(pid, d["day"], True)
                        st.rerun()
        with t2:
            n = a.get("notes", {})
            st.markdown(n.get("summary", ""))
            for c in n.get("key_concepts", []):
                st.markdown(f"- **{c.get('term')}** - {c.get('explanation')}")
        with t3:
            qs = a.get("quiz", {}).get("questions", [])
            answers = {i: st.radio(q["question"], q["options"], index=None, key=f"q{i}") for i, q in enumerate(qs)}
            if st.button("Check answers") and qs:
                score = sum(1 for i, q in enumerate(qs) if answers[i] is not None and q["options"].index(answers[i]) == q["answer_index"])
                pct = round(100 * score / len(qs))
                st.success(f"Score: {score}/{len(qs)} ({pct}%) - saved to your progress")
                agent.memory.update_progress(pid, max(1, prog.get("days_completed", 0)), True, pct)
        st.download_button("⬇️ Download study pack", render_study_markdown(res), file_name=Path(res["saved_files"].get("markdown", "study.md")).name)
        st.caption(f"Auto-saved: `{res['saved_files'].get('markdown')}`")
