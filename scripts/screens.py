"""Screenshot helpers for the report (brief section 12: screenshots/).

* `terminal_png(text, path, title)`  - renders real command output as a terminal-style PNG
                                       (Colab has no desktop, so this is how terminal output is captured).
* `streamlit_png(url, path)`         - real browser screenshot of the Streamlit UI (Playwright, optional).
* `architecture_png(path)`           - the agent graph diagram.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def _font(bold=False, size=15):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def terminal_png(text: str, path: str | Path, title: str = "Terminal", width_chars: int = 110, max_lines: int = 60) -> Path:
    """Render text as a dark terminal window. Long lines are wrapped, very long output is cut."""
    lines = []
    for raw in text.replace("\t", "    ").splitlines():
        lines += textwrap.wrap(raw, width_chars, replace_whitespace=False, drop_whitespace=False) or [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"... ({len(lines) - max_lines + 1} more lines)"]
    f, fb = _font(), _font(True)
    cw, lh = f.getbbox("M")[2], 21
    W, H = 30 + cw * width_chars, 56 + lh * len(lines) + 20
    img = Image.new("RGB", (W, H), (30, 32, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=(52, 55, 66))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 22, 11, 26 + i * 22, 23], fill=c)
    d.text((90, 8), title, font=fb, fill=(220, 220, 220))
    y = 48
    for ln in lines:
        s = ln.strip()
        color = (220, 223, 228)
        if s.startswith("$") or s.startswith(">>>"):
            color = (126, 231, 135)
        elif s.startswith("[Tool Called]") or s.startswith("[Plan]"):
            color = (121, 192, 255)
        elif s.startswith("[Agent Step]") or s.startswith("[Revision]"):
            color = (255, 203, 107)
        elif s.startswith("[Tool Failed]") or "ERROR" in s:
            color = (255, 123, 114)
        elif s.startswith("===") or s.isupper() and len(s) > 3:
            color = (199, 146, 234)
        d.text((15, y), ln, font=f, fill=color)
        y += lh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def streamlit_png(url: str, path: str | Path, wait_text: str = "Review score", timeout_s: int = 900) -> Path | None:
    """Open the Streamlit app in headless Chromium and screenshot it once the result is shown."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed - skipping Streamlit screenshot")
        return None
    path = Path(path)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 1700})
            pg.goto(url, timeout=120_000)
            pg.get_by_text(wait_text).first.wait_for(timeout=timeout_s * 1000)
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(path), full_page=True)
            b.close()
        return path
    except Exception as e:                                         # screenshots must never break the notebook
        print("Streamlit screenshot failed:", str(e)[:200])
        return None


def architecture_png(path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=150)
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
    C = {"io": "#E8F0FE", "node": "#FFF4E5", "llm": "#E6F4EA", "store": "#F3E8FD", "edge": "#5F6368"}

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#3C4043", lw=1.1))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.8, color="#3C4043")

    def arrow(x1, y1, x2, y2, label="", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=C["edge"], lw=1.3,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=7.8, color="#B3261E", style="italic")

    ax.text(6.5, 6.95, "TrendPilot - LangGraph agent workflow (local Ollama · Gemma 3:4B)", ha="center", fontsize=13, weight="bold")
    box(0.2, 5.3, 2.0, 1.0, "User input", "topic · platform · tone\noutput type", C["io"])
    box(2.7, 5.3, 2.0, 1.0, "Memory", "recall similar runs +\nlearned preferences", C["store"])
    box(5.2, 5.3, 2.0, 1.0, "Research", "DuckDuckGo trends\n(optional)", C["node"])
    box(7.7, 5.3, 2.2, 1.0, "Understand", "LLM: audience, goal,\nkey points (JSON)", C["llm"])
    box(10.4, 5.3, 2.4, 1.0, "Planner", "LLM picks tools from\nregistry -> validated plan", C["llm"])
    arrow(2.2, 5.8, 2.7, 5.8); arrow(4.7, 5.8, 5.2, 5.8); arrow(7.2, 5.8, 7.7, 5.8); arrow(9.9, 5.8, 10.4, 5.8)

    box(3.0, 2.75, 7.0, 1.75, "", "", C["node"])
    ax.text(6.5, 4.25, "Execute tools (each prints [Tool Called])", ha="center", fontsize=10.5, weight="bold")
    tools = ["Trend Idea\nGenerator", "Caption\nWriter", "Hashtag\nGenerator", "Reel Script\nGenerator", "Title &\nThumbnail", "Image\nPrompt"]
    for i, t in enumerate(tools):
        ax.add_patch(FancyBboxPatch((3.2 + i * 1.13, 2.95), 1.0, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec="#9AA0A6"))
        ax.text(3.7 + i * 1.13, 3.42, t, ha="center", va="center", fontsize=7.6)
    arrow(11.6, 5.3, 9.6, 4.5, rad=-0.2)

    box(10.5, 2.95, 2.3, 1.35, "Content Reviewer", "LLM rubric (5 scores)\n+ platform rule checks", C["llm"])
    arrow(10.0, 3.6, 10.5, 3.6)
    arrow(11.2, 2.95, 7.0, 2.75, rad=-0.22)
    ax.text(8.9, 2.0, "score < 7.5 or rule broken ->\nre-run only the flagged tools (max 2x)", ha="center", va="center",
            fontsize=7.8, color="#B3261E", style="italic")

    box(10.5, 0.55, 2.3, 1.2, "File Saver", ".md post / script\n+ full .json result", C["store"])
    box(7.6, 0.55, 2.4, 1.2, "Memory save", "SQLite: run, score,\npreferences", C["store"])
    box(4.7, 0.55, 2.4, 1.2, "Final output", "idea · audience · hook ·\ncaption · tags · script", C["io"])
    box(0.3, 0.55, 3.8, 1.2, "User feedback loop", "\"make it funnier\" -> LLM router\nre-runs only the affected tools", C["io"])
    arrow(11.65, 2.95, 11.65, 1.75, "ready")
    arrow(10.5, 1.15, 10.0, 1.15); arrow(7.6, 1.15, 7.1, 1.15); arrow(4.7, 1.15, 4.1, 1.15)
    arrow(2.2, 1.75, 3.6, 2.85, rad=-0.2)
    ax.text(0.2, 0.15, "Demos: CLI (demo.py) · Streamlit (app/streamlit_app.py) · FastAPI (app/main.py)  ·  "
            "StudyBuddy mode reuses the same graph with Study Planner, Notes, Quiz and Progress tools", fontsize=8, color="#5F6368")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "streamlit":     # run outside Jupyter's asyncio loop
        sys.exit(0 if streamlit_png(sys.argv[2], sys.argv[3]) else 1)
    out = Path(os.environ.get("OUT", "screenshots"))
    print(architecture_png(out / "architecture.png"))
