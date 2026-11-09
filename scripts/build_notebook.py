"""Generates notebooks/TrendPilot_Colab.ipynb (run: python scripts/build_notebook.py)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# 🚀 TrendPilot - Local Agentic AI Content Planner (+ StudyBuddy mode)
**AIRI Team PITB - AI Internship Task 2 · Hamza Nadeem**

An end-to-end **agent** (not a chatbot) running on a **local LLM (Gemma 3:4B via Ollama)**. Give it a topic, platform and tone - it
understands the request, **plans which tools to use**, calls them (idea generator, caption writer, hashtag generator, reel script,
title/thumbnail, reviewer, file saver, memory), **reviews and rewrites its own weak parts**, saves the result and remembers it.

### How to run (once)
1. Upload **`agentic_ai_task_2.zip`** to the main folder of your Google Drive (**My Drive**). Do not unzip it.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**. Allow Google Drive access when asked.
4. Wait ~35-45 minutes (model download + 20 test runs). Everything is saved back to `My Drive/agentic_ai_task_2/`.

At the end you get: screenshots, 10 test results (v1 vs v2), the error-analysis table, `report/final_report.pdf`,
the LinkedIn report and a ready-to-upload `agentic_ai_task_2_final.zip`.
""")

md("## Step 1 - Check the GPU")
code(r"""
import subprocess
gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "No GPU found - Runtime > Change runtime type > T4 GPU (CPU works but is ~10x slower)")
""")

md("## Step 2 - Get the project from Google Drive")
code(r"""
import os, sys, shutil, zipfile, json, time
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
WORK = Path(os.environ.get("TP_PROJECT_DIR", "/content/agentic_ai_task_2"))     # fast local copy we work in
DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_DIR = DRIVE_ROOT / "agentic_ai_task_2"                                      # results are copied back here

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

def unpack(zip_path, dest):
    tmp = Path("/content/_unzip"); shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(zip_path).extractall(tmp)
    inner = next((p.parent for p in tmp.rglob("demo.py") if "__MACOSX" not in str(p)), tmp)
    shutil.copytree(inner, dest)

if not (WORK / "app" / "agent.py").exists():
    if (DRIVE_DIR / "app" / "agent.py").exists():
        shutil.copytree(DRIVE_DIR, WORK, dirs_exist_ok=True)                          # continue from a previous session
    elif (DRIVE_ROOT / "agentic_ai_task_2.zip").exists():
        unpack(DRIVE_ROOT / "agentic_ai_task_2.zip", WORK)
    elif IN_COLAB:
        from google.colab import files
        print("Zip not found in My Drive - please upload agentic_ai_task_2.zip now")
        up = files.upload(); unpack(next(iter(up)), WORK)
    else:
        raise FileNotFoundError("Project not found")

os.chdir(WORK); sys.path.insert(0, str(WORK))
for d in ["screenshots", "report", "tests", "outputs/logs"]:
    (WORK / d).mkdir(parents=True, exist_ok=True)
print("Project folder:", WORK)
print(sorted(p.name for p in WORK.iterdir()))
""")

md("## Step 3 - Install Python packages")
code(r"""
!pip install -q -r requirements.txt
print("Packages installed")
""")

md(r"""
## Step 4 - Install Ollama, start the server and download Gemma 3:4B
Ollama runs the model **locally on the Colab GPU** - no paid API. The model download is ~3.3 GB (2-4 minutes).
""")
code(r"""
import requests, subprocess, time, os
MODEL = os.environ.get("TRENDPILOT_MODEL", "gemma3:4b")
OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")

def ollama_up():
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=3).ok
    except Exception:
        return False

if not ollama_up():
    if shutil.which("ollama") is None:
        !sudo apt-get -qq install -y zstd pciutils lshw > /dev/null 2>&1
        !curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1
    env = dict(os.environ, OLLAMA_KEEP_ALIVE="30m")
    subprocess.Popen(["ollama", "serve"], stdout=open("/content/ollama.log", "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        if ollama_up(): break
        time.sleep(1)
print("Ollama server running:", ollama_up())

t0 = time.time()
!ollama pull {MODEL} 2>&1 | tail -n 2
print(f"Model ready in {time.time()-t0:.0f}s")
""")

md(r"""
## Step 5 - Test the model (brief section 10, step 3) → `screenshots/ollama_running.png`
Runs `ollama --version`, `ollama list` and the test prompt from the brief, then saves the real terminal output as a screenshot.
""")
code(r"""
import re
from scripts.screens import terminal_png

def sh(cmd, timeout=600):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    text = (out.stdout + out.stderr)
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)                   # remove terminal colour/spinner codes
    text = re.sub(r"[⠀-⣿]", "", text)                                    # remove spinner characters
    return "\n".join(l.rstrip() for l in text.splitlines()).strip()

TEST_PROMPT = "Give me 5 viral LinkedIn post ideas about AI internships."
log = []
for cmd in ["ollama --version", "ollama list", f'ollama run {MODEL} "{TEST_PROMPT}"']:
    t0 = time.time()
    res = sh(cmd)
    log.append(f"$ {cmd}\n{res}" + (f"\n[{time.time()-t0:.1f}s]" if "run" in cmd else ""))
    print(log[-1], "\n")
gpu_line = sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader")
log.append(f"$ nvidia-smi --query-gpu=name,memory.used,memory.total\n{gpu_line}")
terminal_png("\n\n".join(log), "screenshots/ollama_running.png", f"Google Colab - Ollama running {MODEL} locally", max_lines=70)
OLLAMA_VERSION = sh("ollama --version").split()[-1] if sh("ollama --version") else "?"
print("Saved screenshots/ollama_running.png")
""")
code(r"""
from IPython.display import Image, display
display(Image("screenshots/ollama_running.png", width=900))
""")

md("## Step 6 - Basic Python connection to Ollama (brief section 11)")
code(r"""
import requests

def ask_ollama(prompt, model=MODEL):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]

result = ask_ollama("Give me 5 viral AI project ideas for LinkedIn.")
print(result)
""")

md(r"""
## Step 7 - The agent and its tools
`app/tools.py` has a **tool registry**. The planner (LLM) chooses tools from this list for every request - a reel request gets
the script tool, a LinkedIn post gets caption + hashtags, a StudyBuddy request gets the study tools.
""")
code(r"""
import pandas as pd
pd.set_option("display.max_colwidth", 120)
from app.config import Settings
from app.agent import TrendPilotAgent
from app.tools import REGISTRY

settings = Settings()
agent = TrendPilotAgent(settings)
print("Model in use:", agent.llm.ensure_model(), "| LangGraph:", agent.graph is not None)
pd.DataFrame([{"Tool": t.display_name, "Mode": t.mode, "Uses LLM": t.uses_llm, "What it does": t.description}
              for t in REGISTRY.values()])
""")

md(r"""
## Step 8 - Main example from the brief (CLI demo) → `screenshots/demo_output.png`
Runs the real CLI (`python demo.py ...`). Watch the `[Tool Called]` lines: the agent plans, calls tools, reviews itself and saves.
""")
code(r"""
EXAMPLE = dict(topic="I want to create a viral LinkedIn post about my YOLOv8 helmet detection project",
               platform="LinkedIn", tone="Professional but exciting", output="Post + Reel Script")
cmd = [sys.executable, "-u", "demo.py", "--topic", EXAMPLE["topic"], "--platform", EXAMPLE["platform"],
       "--tone", EXAMPLE["tone"], "--output", EXAMPLE["output"]]
before = set((WORK / "outputs/saved_results").glob("*.json"))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
cli_lines = []
for line in proc.stdout:
    print(line, end=""); cli_lines.append(line.rstrip())
proc.wait()
cli_text = "$ python demo.py --topic \"...YOLOv8 helmet detection project\" --platform LinkedIn --tone \"Professional but exciting\" --output \"Post + Reel Script\"\n" + "\n".join(cli_lines)
terminal_png(cli_text, "screenshots/demo_output.png", "TrendPilot CLI demo - python demo.py", max_lines=95)
example_json = max(set((WORK / "outputs/saved_results").glob("*.json")) - before, key=lambda p: p.stat().st_mtime)
res = json.loads(example_json.read_text())
SHOW = {"model": res["model"], "ollama_version": OLLAMA_VERSION, "gpu": gpu or "CPU", "example_json": str(example_json)}
json.dump(SHOW, open("outputs/logs/showcase.json", "w"), indent=2)
print("\nSaved screenshots/demo_output.png and", example_json.name)
""")
code(r"""
from IPython.display import Markdown
display(Image("screenshots/demo_output.png", width=950))
display(Markdown(Path(res["final_output"]["saved_file"]).read_text()))
""")

md(r"""
## Step 9 - User feedback loop
The user asks for a change; an LLM router decides **which tools to re-run** (only those), the reviewer checks again and a new
`_revised` file is saved.
""")
code(r"""
FEEDBACK = "Make the hook shorter and more surprising, and add one line about what I learned from the project."
revised = agent.revise(res, FEEDBACK)
print("\nBEFORE hook:", res["artifacts"].get("hook"))
print("AFTER  hook:", revised["artifacts"].get("hook"))
print("Re-ran tools:", revised["revised_tools"], "| score", res["review"].get("overall"), "->", revised["review"].get("overall"))
print("Saved:", revised["saved_files"].get("markdown"))
json.dump(revised, open("outputs/logs/revised_example.json", "w"), indent=2, default=str)
SHOW.update(revised_json="outputs/logs/revised_example.json", feedback=FEEDBACK)
json.dump(SHOW, open("outputs/logs/showcase.json", "w"), indent=2)
""")

md(r"""
## Step 10 - Memory
Long-term memory is SQLite (`outputs/memory.db`): past runs, learned preferences and study progress. Before every run the
**Simple Memory Tool** recalls similar topics so the agent does not repeat the same hook.
""")
code(r"""
print("Learned preferences:", agent.memory.prefs())
print("\nWhat the agent will remember next time it writes about YOLOv8:\n")
print(agent.memory.memory_notes("YOLOv8 helmet detection project", "LinkedIn"))
pd.DataFrame(agent.memory.history(10))
""")

md("## Step 11 - StudyBuddy mode (same agent engine, different tools)")
code(r"""
study = agent.run_study("Object detection with YOLOv8", days=5, level="Beginner", hours_per_day=2)
pid = study["artifacts"].get("plan_id")
quiz = study["artifacts"].get("quiz", {}).get("questions", [])
print("\nQuiz example:", quiz[0]["question"] if quiz else "-")
print("Progress after finishing day 1 with 80% quiz score:", agent.memory.update_progress(pid, 1, True, 80) if pid else "-")
json.dump(study, open("outputs/logs/study_example.json", "w"), indent=2, default=str)
SHOW.update(study_json="outputs/logs/study_example.json")
json.dump(SHOW, open("outputs/logs/showcase.json", "w"), indent=2)
display(Markdown(Path(study["saved_files"]["markdown"]).read_text()[:3500]))
""")

md(r"""
## Step 12 - Testing: the 10 test cases from the brief (v1 baseline) ⏱ ~8-10 min
**v1** = simple prompts, fixed tool pipeline, no review loop. Its outputs go to `outputs/baseline_v1/` so they don't mix with the
final outputs.
""")
code(r"""
from app.evaluation import TEST_CASES, run_suite, compare, error_analysis, write_test_cases_md

s1 = Settings(); s1.prompt_version = "v1"; s1.verbose = False
s1.outputs_dir = WORK / "outputs" / "baseline_v1"; s1.memory_db = s1.outputs_dir / "memory_v1.db"
t0 = time.time()
v1 = run_suite(TrendPilotAgent(s1), label="v1", out_dir=WORK / "outputs" / "logs")
print(f"v1 done in {(time.time()-t0)/60:.1f} min")
""")

md("## Step 13 - Testing: the same 10 cases with the final agent (v2) ⏱ ~20-25 min")
code(r"""
s2 = Settings(); s2.verbose = False
t0 = time.time()
v2 = run_suite(TrendPilotAgent(s2), label="v2", out_dir=WORK / "outputs" / "logs")
print(f"v2 done in {(time.time()-t0)/60:.1f} min")
pd.DataFrame(compare(v1, v2))
""")

md(r"""
## Step 14 - Error analysis + `tests/test_cases.md`
Weak/failed answers are detected automatically (platform rules + reviewer score). Read some outputs yourself and, if you want,
add your own observations to `MANUAL_REVIEW_NOTES` (test number → note); they go into the report.
""")
code(r"""
MANUAL_REVIEW_NOTES = {
    # 5: "Humour felt forced - small 4B models are weaker at jokes than at technical posts.",
}
errors = error_analysis(v1, v2, manual_notes=MANUAL_REVIEW_NOTES)
write_test_cases_md("tests/test_cases.md", v1, v2, errors, res["model"])
SHOW["manual_notes"] = MANUAL_REVIEW_NOTES
json.dump(SHOW, open("outputs/logs/showcase.json", "w"), indent=2)
pd.DataFrame(errors).drop(columns=["code"])
""")
code(r"""
pd.DataFrame([{"#": r["no"], "Input": r.get("input_prompt"), "Tools": len(r.get("tools_used", [])), "Score": r.get("review_score"),
               "Useful": r.get("useful"), "Hook / first line": r.get("preview"), "Saved file": r.get("saved_file")} for r in v2])
""")

md("## Step 15 - Saved files → `screenshots/saved_file_output.png`")
code(r"""
listing = []
for sub in ["generated_posts", "generated_scripts", "saved_results", "study_plans"]:
    files_ = sorted((WORK / "outputs" / sub).glob("*"))
    listing.append(f"outputs/{sub}/  ({len(files_)} files)")
    listing += [f"    {f.name}" for f in files_[:6]] + ([f"    ... {len(files_)-6} more"] if len(files_) > 6 else [])
md_file = Path(res["final_output"]["saved_file"])
content = md_file.read_text().splitlines()
text = "$ ls outputs/*\n" + "\n".join(listing) + f"\n\n$ cat {md_file.relative_to(WORK)}\n" + "\n".join(content[:38])
terminal_png(text, "screenshots/saved_file_output.png", "Saved agent outputs", max_lines=80)
display(Image("screenshots/saved_file_output.png", width=900))
""")

md("## Step 16 - Architecture diagram")
code(r"""
from scripts.screens import architecture_png
architecture_png("screenshots/architecture.png")
display(Image("screenshots/architecture.png", width=950))
""")

md(r"""
## Step 17 - FastAPI demo (Option C)
Starts the API in the background and calls it like any client would. (Docs at `/docs` when run locally.)
""")
code(r"""
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
                       stdout=open("/content/api.log" if IN_COLAB else os.devnull, "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        if requests.get("http://localhost:8000/health", timeout=2).ok: break
    except Exception: time.sleep(1)
print("GET /health ->", requests.get("http://localhost:8000/health").json())
r = requests.post("http://localhost:8000/generate-content", timeout=900,
                  json={"topic": "Debugging FastAPI at 2am", "platform": "X / Twitter", "tone": "Funny", "output_type": "Short post"})
api_out = r.json()
print("POST /generate-content ->", r.status_code)
print(json.dumps({k: api_out[k] for k in ["hook", "caption", "hashtags", "review_score", "tools_used", "saved_file"]}, indent=2))
api.terminate()
""")

md(r"""
## Step 18 - Streamlit demo (Option B)
Starts the Streamlit app, takes a real browser screenshot of it, and (optional) gives you a **public link** to try it yourself.
""")
code(r"""
MAKE_STREAMLIT_SCREENSHOT = True
SHOW_PUBLIC_LINK = True

st_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                            "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                           stdout=open("/content/streamlit.log" if IN_COLAB else os.devnull, "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        if requests.get("http://localhost:8501/_stcore/health", timeout=2).ok: break
    except Exception: time.sleep(1)
print("Streamlit running on port 8501")

if MAKE_STREAMLIT_SCREENSHOT:
    try:
        import playwright  # noqa
    except ImportError:
        !pip install -q playwright && playwright install --with-deps chromium > /dev/null 2>&1
    from urllib.parse import quote
    url = "http://localhost:8501/?demo=1&topic=" + quote("My YOLOv8 safety helmet detection project")
    print("Generating inside Streamlit and taking a screenshot (1-3 min)...")
    shot = subprocess.run([sys.executable, "scripts/screens.py", "streamlit", url, "screenshots/streamlit_demo.png"],
                          capture_output=True, text=True)
    if shot.returncode == 0:
        display(Image("screenshots/streamlit_demo.png", width=950))
    else:
        print("Screenshot skipped:", (shot.stdout + shot.stderr)[-300:])
""")
code(r"""
if SHOW_PUBLIC_LINK and IN_COLAB:
    if not Path("/content/cloudflared").exists():
        !wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared
    tunnel = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://localhost:8501"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            print("Open the Streamlit demo here:", m.group(0)); break
""")

md("## Step 19 - Final report (PDF), LinkedIn report and README results")
code(r"""
agent.memory.export_json("outputs/memory_export.json")
!python scripts/build_report.py
display(Markdown(Path("report/linkedin_portfolio_report.md").read_text()))
""")

md("## Step 20 - Deliverables checklist (brief section 20)")
code(r"""
saved = list((WORK/"outputs/generated_posts").glob("*.md")) + list((WORK/"outputs/generated_scripts").glob("*.md"))
checks = {
    "Complete source code (app/)": all((WORK/"app"/f).exists() for f in ["main.py", "agent.py", "tools.py", "memory.py", "prompts.py"]),
    "requirements.txt": (WORK/"requirements.txt").exists(),
    "README file": (WORK/"README.md").exists(),
    "Working demo (CLI + Streamlit + FastAPI)": (WORK/"demo.py").exists() and (WORK/"app/streamlit_app.py").exists(),
    "Screenshot: Ollama running": (WORK/"screenshots/ollama_running.png").exists(),
    "Screenshot: agent output (demo)": (WORK/"screenshots/demo_output.png").exists(),
    "Screenshot: saved file output": (WORK/"screenshots/saved_file_output.png").exists(),
    f"At least 5 saved outputs (found {len(saved)})": len(saved) >= 5,
    "Test results for 10 inputs": (WORK/"tests/test_cases.md").exists() and len(v2) == 10,
    "Error-analysis table (>=5 rows)": len(errors) >= 5,
    "Final report PDF": (WORK/"report/final_report.pdf").exists(),
    "LinkedIn-ready report section": (WORK/"report/linkedin_portfolio_report.md").exists(),
}
for k, ok in checks.items():
    print(("✅ " if ok else "❌ ") + k)
""")

md(r"""
## Step 21 - Save everything to Google Drive + make the final zip
Copies the whole project (with outputs, screenshots and report) to `My Drive/agentic_ai_task_2/` and creates
`My Drive/agentic_ai_task_2_final.zip`. For GitHub: download that zip, unzip it, and upload the **contents** of the
`agentic_ai_task_2` folder (the folder that directly contains `README.md`).
""")
code(r"""
for p in WORK.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
if IN_COLAB:
    shutil.copytree(WORK, DRIVE_DIR, dirs_exist_ok=True)
    print("Saved to", DRIVE_DIR)
zip_base = (DRIVE_ROOT if IN_COLAB else WORK.parent) / "agentic_ai_task_2_final"
shutil.make_archive(str(zip_base), "zip", WORK.parent, WORK.name)
print("Final zip:", f"{zip_base}.zip")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "TrendPilot_Colab.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("Wrote", out, len(cells), "cells")
