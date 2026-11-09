"""TrendPilot agent - a LangGraph state machine with planner, tool executor, reviewer loop and saver.

    START
      |-> memory      (Memory Tool: recall previous topics/styles)
      |-> research    (Web Trend Research - TrendPilot mode only, optional)
      |-> understand  (LLM: audience, key message, concrete details, things to avoid)
      |-> plan        (PLANNER agent: picks tools from the registry; plan is validated/fixed)
      |-> execute     (calls each tool, logs "[Tool Called]: ...")
      |-> review      (REVIEWER agent: LLM rubric + hard quality rules)
      |      |-- needs_revision & rounds left --> refine (re-run only the flagged tools with feedback) --> review
      |-> save        (File Saver + Memory save)
    END

The same engine runs two agents: TrendPilot (viral content plan, brief Option A) and StudyBuddy
(study plan + notes + quiz + progress, brief Option B).
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, TypedDict

from . import prompts as P
from .config import Settings
from .llm import OllamaClient, OllamaError
from .memory import Memory
from .quality import core_topic, parts_for
from .tools import REGISTRY, ToolContext, catalog

try:
    from langgraph.graph import END, START, StateGraph
    HAS_LANGGRAPH = True
except Exception:                                   # the agent still works without LangGraph (simple loop)
    HAS_LANGGRAPH = False

CONTENT_ORDER = ["trend_idea_generator", "caption_writer", "hashtag_generator", "reel_script_generator",
                 "title_thumbnail_generator", "image_prompt_generator", "content_reviewer", "file_saver"]
PART_TOOL = {"caption": "caption_writer", "hashtags": "hashtag_generator", "reel_script": "reel_script_generator",
             "titles": "title_thumbnail_generator"}
STUDY_PLAN = ["study_planner", "notes_generator", "quiz_generator", "study_reviewer", "progress_saver", "file_saver"]
FEEDBACK_KEYWORDS = {
    "hashtag_generator": ("hashtag", "tags"),
    "reel_script_generator": ("script", "reel", "video", "scene", "voiceover", "seconds"),
    "title_thumbnail_generator": ("title", "thumbnail"),
    "caption_writer": ("hook", "caption", "post", "shorter", "longer", "tone", "funny", "emoji", "cta", "professional"),
}


class AgentState(TypedDict, total=False):
    ctx: Any
    mode: str
    plan: list
    plan_fixes: list
    revisions: int
    trace: list


class TrendPilotAgent:
    def __init__(self, settings: Settings | None = None, llm: OllamaClient | None = None,
                 memory: Memory | None = None, on_event: Callable[[dict], None] | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_dirs()
        self.llm = llm or OllamaClient(self.settings)
        self.memory = memory or Memory(self.settings.memory_db)
        self.on_event = on_event
        self.graph = self._build_graph() if HAS_LANGGRAPH else None

    # ================================================================== logging
    def _emit(self, state: dict, kind: str, name: str, detail: Any = None, seconds: float = 0.0, ok: bool = True) -> None:
        ev = {"t": time.strftime("%H:%M:%S"), "kind": kind, "name": name, "detail": detail, "seconds": round(seconds, 2), "ok": ok}
        state.setdefault("trace", []).append(ev)
        if self.settings.verbose:
            label = {"tool": "[Tool Called]", "step": "[Agent Step]", "plan": "[Plan]", "review": "[Review]",
                     "revise": "[Revision]", "error": "[Tool Failed]", "skip": "[Tool Skipped]"}.get(kind, f"[{kind}]")
            extra = f" -> {detail}" if detail not in (None, "", {}) else ""
            print(f"{label}: {name}{extra}" + (f"  ({seconds:.1f}s)" if seconds else ""), flush=True)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    def _call_tool(self, state: dict, name: str) -> bool:
        t = REGISTRY[name]
        t0 = time.time()
        try:
            out = t.run(state["ctx"])
        except Exception as e:            # one failing tool must not crash the whole agent run
            self._emit(state, "error", t.display_name, f"{type(e).__name__}: {str(e)[:160]}", time.time() - t0, ok=False)
            return False
        kind = "skip" if isinstance(out, dict) and "skipped" in out else "tool"
        self._emit(state, kind, t.display_name, out, time.time() - t0)
        return True

    # ================================================================== graph nodes
    def n_memory(self, s):
        self._call_tool(s, "memory_tool")
        return {"trace": s["trace"]}

    def n_research(self, s):
        self._call_tool(s, "trend_research")
        return {"trace": s["trace"]}

    def n_understand(self, s):
        ctx: ToolContext = s["ctx"]
        t0 = time.time()
        try:
            u = ctx.llm.generate_json(P.understand_prompt(ctx.request, ctx.artifacts.get("memory_notes"), ctx.artifacts.get("research"), ctx.v),
                                      ctx.system, P.SCHEMAS["understand"], tool="understand_goal", temperature=0.3, max_tokens=500)
        except OllamaError as e:
            u = {}
            self._emit(s, "error", "Understand goal", str(e)[:120], time.time() - t0, ok=False)
        ctx.artifacts["understanding"] = u
        if u.get("target_audience"):
            ctx.artifacts.setdefault("target_audience", u["target_audience"])
        if u:
            self._emit(s, "step", "Understand goal", {"audience": u.get("target_audience", "")[:80]}, time.time() - t0)
        return {"trace": s["trace"]}

    def n_plan(self, s):
        ctx: ToolContext = s["ctx"]
        if s["mode"] == "study":
            plan = [{"tool": t, "reason": REGISTRY[t].description} for t in STUDY_PLAN]
            self._emit(s, "plan", "Study plan", " -> ".join(STUDY_PLAN))
            return {"plan": plan, "plan_fixes": [], "trace": s["trace"]}
        needed_tools = [PART_TOOL[p] for p in parts_for(ctx.request["output_type"], ctx.request["platform"])]
        full = "full" in ctx.request["output_type"].lower() or "plan" in ctx.request["output_type"].lower()
        raw, fixes = [], []
        t0 = time.time()
        if ctx.v == "v2":
            try:
                out = ctx.llm.generate_json(P.planner_prompt(ctx.request, catalog("content"), ctx.artifacts.get("understanding")),
                                            ctx.system, P.SCHEMAS["planner"], tool="planner", temperature=0.2, max_tokens=500)
                raw = [st for st in out.get("steps", []) if isinstance(st, dict)]
            except OllamaError as e:
                fixes.append(f"planner failed ({str(e)[:60]}) - used default plan")
        else:
            fixes.append("v1 baseline uses a fixed pipeline (no LLM planner)")
        plan, seen = [], set()
        for st in raw:
            name = str(st.get("tool", "")).strip()
            if name in ("memory_tool", "trend_research"):
                continue                                     # already executed before planning
            if name not in REGISTRY or REGISTRY[name].mode == "study":
                fixes.append(f"removed unknown tool '{name}'")
                continue
            if name in PART_TOOL.values() and name not in needed_tools:
                fixes.append(f"removed '{name}' (not requested: {ctx.request['output_type']})")
                continue
            if name == "image_prompt_generator" and not full:
                fixes.append("removed 'image_prompt_generator' (only for full content plans)")
                continue
            if name not in seen:
                seen.add(name)
                plan.append({"tool": name, "reason": st.get("reason", "")})
        for name in ["trend_idea_generator"] + needed_tools + (["image_prompt_generator"] if full else []) + ["content_reviewer", "file_saver"]:
            if name not in seen:
                if raw:
                    fixes.append(f"added missing '{name}'")
                plan.append({"tool": name, "reason": REGISTRY[name].description})
                seen.add(name)
        plan.sort(key=lambda st: CONTENT_ORDER.index(st["tool"]) if st["tool"] in ("trend_idea_generator", "content_reviewer", "file_saver")
                  else CONTENT_ORDER.index("caption_writer") + 0.5)    # idea first, reviewer + saver last, keep LLM order between
        self._emit(s, "plan", "Planner agent", " -> ".join(st["tool"] for st in plan), time.time() - t0)
        for f in fixes:
            self._emit(s, "plan", "Plan check", f)
        return {"plan": plan, "plan_fixes": fixes, "trace": s["trace"]}

    def n_execute(self, s):
        for st in s["plan"]:
            if st["tool"] not in ("content_reviewer", "study_reviewer", "file_saver", "progress_saver"):
                self._call_tool(s, st["tool"])
        return {"trace": s["trace"]}

    def n_review(self, s):
        name = "study_reviewer" if s["mode"] == "study" else "content_reviewer"
        ok = self._call_tool(s, name)
        if not ok:
            s["ctx"].artifacts["review"] = {"overall": None, "verdict": "review_failed", "issues": [], "suggestions": [], "rewrite": []}
        return {"trace": s["trace"]}

    def route_after_review(self, s) -> str:
        rev = s["ctx"].artifacts.get("review", {})
        if rev.get("verdict") == "needs_revision" and rev.get("rewrite") and s.get("revisions", 0) < self.settings.max_revisions:
            return "refine"
        return "save"

    def n_refine(self, s):
        ctx: ToolContext = s["ctx"]
        rev = ctx.artifacts["review"]
        n = s.get("revisions", 0) + 1
        self._emit(s, "revise", f"Round {n}", f"re-running {rev['rewrite']} (score {rev['overall']})")
        for name in rev["rewrite"]:
            mine = [i["message"] for i in rev.get("rule_issues", []) if i["tool"] == name]
            ctx.feedback[name] = "\n".join(f"- {x}" for x in mine + rev.get("issues", [])[:4] + rev.get("suggestions", [])[:4])
            self._call_tool(s, name)
        ctx.feedback.clear()
        ctx.artifacts.setdefault("review_history", []).append({k: rev.get(k) for k in ("overall", "verdict", "issues", "rewrite")})
        return {"revisions": n, "trace": s["trace"]}

    def n_save(self, s):
        ctx: ToolContext = s["ctx"]
        ctx.artifacts["_result_snapshot"] = self._result(s, partial=True)
        if s["mode"] == "study":
            self._call_tool(s, "progress_saver")
            ctx.artifacts["_result_snapshot"] = self._result(s, partial=True)
        self._call_tool(s, "file_saver")
        if s["mode"] == "study":
            if ctx.artifacts.get("plan_id"):
                self.memory.set_plan_file(ctx.artifacts["plan_id"], ctx.artifacts.get("saved_files", {}).get("markdown", ""))
        else:
            t0 = time.time()
            rid = self.memory.save_run(self._result(s, partial=True) | {"saved_files": ctx.artifacts.get("saved_files", {})})
            self._emit(s, "tool", "Simple Memory Tool (save)", {"run_id": rid}, time.time() - t0)
        return {"trace": s["trace"]}

    # ================================================================== graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        for name in ("memory", "research", "understand", "plan", "execute", "review", "refine", "save"):
            g.add_node(name, getattr(self, f"n_{name}"))
        g.add_edge(START, "memory")
        g.add_conditional_edges("memory", lambda s: "plan" if s["mode"] == "study" else "research",
                                {"plan": "plan", "research": "research"})
        g.add_edge("research", "understand")
        g.add_edge("understand", "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "review")
        g.add_conditional_edges("review", self.route_after_review, {"refine": "refine", "save": "save"})
        g.add_edge("refine", "review")
        g.add_edge("save", END)
        return g.compile()

    def _invoke(self, state: dict) -> dict:
        if self.graph is not None:
            return self.graph.invoke(state, {"recursion_limit": 40})
        # fallback: same flow as a plain loop
        self.n_memory(state)
        if state["mode"] != "study":
            self.n_research(state); self.n_understand(state)
        state.update(self.n_plan(state)); self.n_execute(state); self.n_review(state)
        while self.route_after_review(state) == "refine":
            state.update(self.n_refine(state)); self.n_review(state)
        self.n_save(state)
        return state

    # ================================================================== public API
    def run(self, topic: str, platform: str = "LinkedIn", tone: str = "Professional", output_type: str = "Post + hashtags") -> dict:
        req = {"topic": topic.strip(), "core_topic": core_topic(topic), "platform": platform, "tone": tone, "output_type": output_type}
        return self._run(req, "content")

    def run_study(self, topic: str, days: int = 7, level: str = "Beginner", hours_per_day: float = 2.0, goal: str = "") -> dict:
        req = {"topic": topic.strip(), "days": int(days), "level": level, "hours_per_day": float(hours_per_day), "goal": goal}
        return self._run(req, "study")

    def _run(self, req: dict, mode: str) -> dict:
        t0 = time.time()
        call0 = len(self.llm.calls)
        ctx = ToolContext(self.llm, self.settings, self.memory, req)
        state = {"ctx": ctx, "mode": mode, "plan": [], "plan_fixes": [], "revisions": 0, "trace": []}
        if self.settings.verbose:
            print(f"\n=== {'StudyBuddy' if mode == 'study' else 'TrendPilot'} | {req['topic']} ===", flush=True)
        final = self._invoke(state)
        result = self._result(final)
        result["total_seconds"] = round(time.time() - t0, 1)
        result["stats"] = self.llm.stats(call0)
        self._log_run(result)
        if self.settings.verbose:
            print(f"=== done in {result['total_seconds']}s | review {result['review'].get('overall')} "
                  f"({result['review'].get('verdict')}) | revisions {result['revisions']} | saved: {result['saved_files'].get('markdown')} ===", flush=True)
        return result

    def _result(self, s: dict, partial: bool = False) -> dict:
        ctx: ToolContext = s["ctx"]
        a = {k: v for k, v in ctx.artifacts.items() if not k.startswith("_")}
        tools_used = [e["name"] for e in s.get("trace", []) if e["kind"] == "tool"]
        res = {"mode": s["mode"], "request": ctx.request, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "model": self.llm.model, "prompt_version": self.settings.prompt_version,
               "plan": s.get("plan", []), "plan_fixes": s.get("plan_fixes", []), "artifacts": a,
               "review": a.get("review", {}), "revisions": s.get("revisions", 0), "tools_used": tools_used,
               "trace": s.get("trace", []), "saved_files": a.get("saved_files", {})}
        if not partial:
            res["final_output"] = self.final_output(res)
        return res

    # ------------------------------------------------------------------ user feedback loop
    def revise(self, result: dict, feedback: str) -> dict:
        """Apply user feedback ("make the hook shorter", "funnier", ...) to an existing result."""
        t0 = time.time()
        call0 = len(self.llm.calls)
        mode = result.get("mode", "content")
        ctx = ToolContext(self.llm, self.settings, self.memory, dict(result["request"]),
                          artifacts=copy.deepcopy({k: v for k, v in result["artifacts"].items() if k not in ("saved_files", "review")}))
        state = {"ctx": ctx, "mode": mode, "plan": result.get("plan", []), "plan_fixes": [],
                 "revisions": result.get("revisions", 0), "trace": []}
        if mode == "study":
            allowed = ["study_planner", "notes_generator", "quiz_generator"]
            tools = [t for t, kws in {"quiz_generator": ("quiz", "question"), "notes_generator": ("note", "summary", "concept"),
                                       "study_planner": ("plan", "day", "time", "schedule")}.items() if any(k in feedback.lower() for k in kws)] or ["study_planner"]
        else:
            allowed = [PART_TOOL[p] for p in parts_for(ctx.request["output_type"], ctx.request["platform"])]
            tools = []
            try:
                out = self.llm.generate_json(P.feedback_router_prompt(ctx.request, feedback, allowed), ctx.system,
                                             P.SCHEMAS["feedback_router"], tool="feedback_router", temperature=0.1, max_tokens=200)
                tools = [t for t in out.get("tools", []) if t in allowed]
            except OllamaError:
                pass
            if not tools:      # keyword fallback
                tools = [t for t, kws in FEEDBACK_KEYWORDS.items() if t in allowed and any(k in feedback.lower() for k in kws)] or allowed[:1]
        ctx.artifacts["_is_revision"] = True
        self._emit(state, "revise", "User feedback", f"\"{feedback}\" -> {tools}")
        for t in tools:
            ctx.feedback[t] = f"The user asked: {feedback}"
            self._call_tool(state, t)
        ctx.feedback.clear()
        self.n_review(state)
        state["revisions"] = state.get("revisions", 0) + 1
        self.n_save(state)
        new = self._result(state)
        new["user_feedback"] = feedback
        new["revised_tools"] = tools
        new["total_seconds"] = round(time.time() - t0, 1)
        new["stats"] = self.llm.stats(call0)
        self._log_run(new)
        return new

    # ------------------------------------------------------------------ presentation
    @staticmethod
    def final_output(res: dict) -> dict:
        """The final output fields required by the brief (section 13, step 4)."""
        a, req = res["artifacts"], res["request"]
        if res["mode"] == "study":
            plan = a.get("study_plan", {})
            return {"topic": req["topic"], "plan_title": plan.get("title"), "days": len(plan.get("days", [])),
                    "quiz_questions": len(a.get("quiz", {}).get("questions", [])), "plan_id": a.get("plan_id"),
                    "review": res["review"].get("overall"), "saved_file": a.get("saved_files", {}).get("markdown")}
        idea = a.get("content_idea") or {}
        sc = a.get("reel_script") or {}
        return {
            "topic": req["topic"], "platform": req["platform"], "tone": req["tone"],
            "content_idea": f"{idea.get('title', '')} - {idea.get('angle', '')}".strip(" -"),
            "target_audience": a.get("target_audience", ""),
            "hook": a.get("hook", ""), "caption": a.get("caption", ""), "hashtags": a.get("hashtags", []),
            "reel_script": "\n".join(f"{x.get('time', '')}: [{x.get('visual', '')}] {x.get('voiceover', '')}" for x in sc.get("scenes", [])),
            "title": (a.get("titles") or {}).get("best_title", ""), "thumbnail_text": (a.get("titles") or {}).get("thumbnail_text", ""),
            "review_comments": res["review"].get("suggestions", [])[:4] or res["review"].get("issues", [])[:4],
            "review_score": res["review"].get("overall"),
            "saved_file": a.get("saved_files", {}).get("markdown", ""),
        }

    @staticmethod
    def format_text(res: dict) -> str:
        fo = res.get("final_output") or TrendPilotAgent.final_output(res)
        if res["mode"] == "study":
            return "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in fo.items())
        sc = res["artifacts"].get("reel_script") or {}
        L = [f"Topic: {fo['topic']}", f"Platform: {fo['platform']}", f"Tone: {fo['tone']}", "",
             "Content Idea:", fo["content_idea"], "", "Target Audience:", fo["target_audience"], ""]
        if fo["hook"]:
            L += ["Hook:", fo["hook"], ""]
        if fo["caption"]:
            L += [f"{P.platform_key(fo['platform'])} Caption:", fo["caption"], ""]
        if fo["hashtags"]:
            L += ["Hashtags:", " ".join(fo["hashtags"]), ""]
        if sc.get("scenes"):
            L += [f"Reel Script (~{sc.get('total_seconds', 0):.0f}s):"]
            for i, x in enumerate(sc["scenes"], 1):
                L += [f"Scene {i} ({x.get('time', '')}): {x.get('visual', '')}", f"Voiceover: \"{x.get('voiceover', '')}\""]
            L += [f"Closing: {sc.get('closing_line', '')}", ""]
        if fo["title"]:
            L += [f"Title: {fo['title']}", f"Thumbnail Text: {fo['thumbnail_text']}", ""]
        L += [f"Review ({fo['review_score']}/10):"] + [f"- {c}" for c in fo["review_comments"]] + ["", "Saved File:", fo["saved_file"]]
        return "\n".join(L)

    def _log_run(self, res: dict) -> None:
        """Append one line per run to outputs/logs/agent_runs.jsonl (agent evaluation log)."""
        row = {"time": res["created_at"], "mode": res["mode"], "request": res["request"], "model": res["model"],
               "prompt_version": res["prompt_version"], "review": res["review"].get("overall"),
               "verdict": res["review"].get("verdict"), "revisions": res["revisions"],
               "rule_issues": [i["code"] for i in res["review"].get("rule_issues", [])], "tools": res["tools_used"],
               "seconds": res.get("total_seconds"), "stats": res.get("stats"), "saved": res["saved_files"],
               "feedback": res.get("user_feedback")}
        with open(self.settings.logs_dir / "agent_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
