"""Central configuration for TrendPilot.

Every value can be overridden with an environment variable, so the same code runs on a laptop,
in Google Colab, or behind FastAPI/Streamlit without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


@dataclass
class Settings:
    # ---- local LLM (Ollama) ----
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _env("TRENDPILOT_MODEL", "gemma3:4b"))
    fallback_models: tuple = ("llama3.2:3b", "qwen2.5:3b", "phi3:mini")
    temperature: float = field(default_factory=lambda: _env("TRENDPILOT_TEMPERATURE", 0.7))
    num_ctx: int = field(default_factory=lambda: _env("TRENDPILOT_NUM_CTX", 4096))
    request_timeout: int = field(default_factory=lambda: _env("TRENDPILOT_TIMEOUT", 300))
    json_retries: int = 2

    # ---- agent behaviour ----
    prompt_version: str = field(default_factory=lambda: _env("TRENDPILOT_PROMPT_VERSION", "v2"))  # v1 = baseline, v2 = improved
    review_threshold: float = field(default_factory=lambda: _env("TRENDPILOT_REVIEW_THRESHOLD", 7.5))
    max_revisions: int = field(default_factory=lambda: _env("TRENDPILOT_MAX_REVISIONS", 2))
    enable_web_search: bool = field(default_factory=lambda: _env("TRENDPILOT_WEB_SEARCH", True))
    web_search_timeout: int = 12
    verbose: bool = field(default_factory=lambda: _env("TRENDPILOT_VERBOSE", True))

    # ---- storage ----
    outputs_dir: Path = field(default_factory=lambda: Path(_env("TRENDPILOT_OUTPUTS", str(PROJECT_ROOT / "outputs"))))
    memory_db: Path = field(default_factory=lambda: Path(_env("TRENDPILOT_MEMORY_DB", str(PROJECT_ROOT / "outputs" / "memory.db"))))

    @property
    def posts_dir(self) -> Path:
        return self.outputs_dir / "generated_posts"

    @property
    def scripts_dir(self) -> Path:
        return self.outputs_dir / "generated_scripts"

    @property
    def results_dir(self) -> Path:
        return self.outputs_dir / "saved_results"

    @property
    def study_dir(self) -> Path:
        return self.outputs_dir / "study_plans"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.posts_dir, self.scripts_dir, self.results_dir, self.study_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.memory_db.parent.mkdir(parents=True, exist_ok=True)


PLATFORMS = ["LinkedIn", "Instagram Reels", "TikTok", "YouTube Shorts", "X / Twitter", "Facebook"]
TONES = ["Professional", "Professional but exciting", "Motivational", "Inspirational", "Funny",
         "Humorous", "Meme style", "Short and catchy", "Exciting", "Educational"]
OUTPUT_TYPES = ["Post + hashtags", "Post", "Reel script", "Short post", "Portfolio post", "Video script",
                "Post + Reel Script", "Full content plan"]
