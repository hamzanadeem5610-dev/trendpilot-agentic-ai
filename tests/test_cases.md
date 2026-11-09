# TrendPilot - Test Cases and Results

Model: **gemma3:4b** (local, Ollama) · 10 test inputs from the task brief · each run twice: **v1** = baseline prompts / fixed pipeline / no review loop, **v2** = improved prompts + planner + reviewer loop.

## Final results (v2 - improved agent)

| # | Input prompt | Tools used | Final output (preview) | Saved file | Score | Useful? |
|---|---|---|---|---|---|---|
| 1 | Topic: YOLOv8 helmet detection / Platform: LinkedIn / Tone: Professional / Output: Post + hashtags | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | I trained an AI to spot missing helmets. The hardest part wasn't the model - it was the data. | `outputs/generated_posts/yolov8_helmet_detection_linkedin_20261104_074922_post.md` | 8.55 | Yes |
| 2 | Topic: AI internship experience / Platform: LinkedIn / Tone: Motivational / Output: Post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | Seriously, keep that LinkedIn profile updated – it’s your first impression. | `outputs/generated_posts/ai_internship_experience_linkedin_20261104_074954_post.md` | 8.8 | Yes |
| 3 | Topic: Python learning journey / Platform: Instagram Reels / Tone: Funny / Output: Reel script | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Reel Script -> Hashtag -> Title and Thumbnail Text -> Content Reviewer -> File Saver -> Simple Memory (save) | Stop scrolling! Seriously. Let's code. | `outputs/generated_scripts/python_learning_journey_instagram_re_20261104_075039_script.md` | 8.3 | Yes |
| 4 | Topic: Machine learning project / Platform: X / Twitter / Tone: Short and catchy / Output: Short post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | Forget labeled datasets. Let’s build a massive X (Twitter) one. | `outputs/generated_posts/machine_learning_project_x_twitter_20261104_075112_post.md` | 6.5 | Partly |
| 5 | Topic: Data annotation struggles / Platform: LinkedIn / Tone: Humorous / Output: Post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | That starter assessment… it’s like a digital gatekeeper. 😬 | `outputs/generated_posts/data_annotation_struggles_linkedin_20261104_075144_post.md` | 8.2 | Yes |
| 6 | Topic: Model failed in production / Platform: Instagram Reels / Tone: Meme style / Output: Reel script | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Reel Script -> Hashtag -> Title and Thumbnail Text -> Content Reviewer -> File Saver -> Simple Memory (save) | Your Reel just died? Don’t panic. | `outputs/generated_scripts/model_failed_in_production_instagram_re_20261104_075227_script.md` | 8.3 | Yes |
| 7 | Topic: Final year project / Platform: LinkedIn / Tone: Professional / Output: Portfolio post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | I trained an AI to spot missing helmets. The hardest part wasn’t the model. | `outputs/generated_posts/final_year_project_linkedin_20261104_075302_post.md` | 8.8 | Yes |
| 8 | Topic: Computer vision project / Platform: YouTube Shorts / Tone: Exciting / Output: Video script | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Reel Script -> Hashtag -> Title and Thumbnail Text -> Content Reviewer -> File Saver -> Simple Memory (save) | My AI fails… spectacularly. Building a computer vision project isn't always glamorous. | `outputs/generated_scripts/computer_vision_project_youtube_shor_20261104_075349_script.md` | 8.2 | Yes |
| 9 | Topic: Debugging FastAPI / Platform: X / Twitter / Tone: Funny / Output: Short post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | FastAPI debugging feels like wrestling a Python dragon. 🐉 | `outputs/generated_posts/debugging_fastapi_x_twitter_20261104_075418_post.md` | 8.2 | Yes |
| 10 | Topic: Open-source contribution / Platform: LinkedIn / Tone: Inspirational / Output: Post | Simple Memory -> Web Trend Research (DuckDuckGo) -> Trend Idea -> Caption Writer -> Hashtag -> Content Reviewer -> File Saver -> Simple Memory (save) | My resume looked like a chaotic list of projects – it needed a serious upgrade. | `outputs/generated_posts/open_source_contribution_linkedin_20261104_075453_post.md` | 8.3 | Yes |

## v1 (baseline) vs v2 (improved)

| # | Topic | Platform | v1 score | v1 issues | v2 score | v2 issues | v2 self-revisions |
|---|---|---|---|---|---|---|---|
| 1 | YOLOv8 helmet detection | LinkedIn | 7.65 | 2 | 8.55 | 1 | 0 |
| 2 | AI internship experience | LinkedIn | 7.1 | 2 | 8.8 | 0 | 0 |
| 3 | Python learning journey | Instagram Reels | 7.35 | 2 | 8.3 | 0 | 0 |
| 4 | Machine learning project | X / Twitter | 3.95 | 3 | 6.5 | 1 | 0 |
| 5 | Data annotation struggles | LinkedIn | 5.8 | 2 | 8.2 | 0 | 0 |
| 6 | Model failed in production | Instagram Reels | 8.6 | 0 | 8.3 | 0 | 0 |
| 7 | Final year project | LinkedIn | 6.95 | 3 | 8.8 | 0 | 0 |
| 8 | Computer vision project | YouTube Shorts | 8.3 | 0 | 8.2 | 0 | 0 |
| 9 | Debugging FastAPI | X / Twitter | 8.05 | 1 | 8.2 | 0 | 0 |
| 10 | Open-source contribution | LinkedIn | 9.2 | 0 | 8.3 | 0 | 0 |

## Error analysis (weak / failed responses)

| Test Case | Problem Found | Possible Reason | Fix Applied | Result After Fix |
|---|---|---|---|---|
| #4 Machine learning project (X / Twitter, Short and catchy) | 350 characters incl. hashtags > X / Twitter limit 280. | X/Twitter's 280-character limit (incl. hashtags) was never mentioned | Added the 280-character rule and an automatic character-count check | fixed - score 3.95 -> 6.5 |
| #5 Data annotation struggles (LinkedIn, Humorous) | Caption has only 50 words; LinkedIn target is 90-260. | No hashtag count was given for the platform | Platform-specific hashtag counts + automatic cap in the hashtag tool | fixed - score 5.8 -> 8.2 |
| #7 Final year project (LinkedIn, Professional) | Hashtags are mixed into the caption text. | No hashtag count was given for the platform | Platform-specific hashtag counts + automatic cap in the hashtag tool | fixed - score 6.95 -> 8.8 |
| #2 AI internship experience (LinkedIn, Motivational) | Hashtags are mixed into the caption text. | No hashtag count was given for the platform | Platform-specific hashtag counts + automatic cap in the hashtag tool | fixed - score 7.1 -> 8.8 |
| #3 Python learning journey (Instagram Reels, Funny) | 15 hashtags; Instagram Reels works best with 5-10. | Model returned broken/invalid JSON for a tool | Ollama structured outputs (JSON schema) + automatic retry with the parse error | fixed - score 7.35 -> 8.3 |
| #1 YOLOv8 helmet detection (LinkedIn, Professional) | 14 hashtags; LinkedIn works best with 3-5. | Model returned broken/invalid JSON for a tool | Ollama structured outputs (JSON schema) + automatic retry with the parse error | fixed - score 7.65 -> 8.55 |
| #9 Debugging FastAPI (X / Twitter, Funny) | 3 hashtags; X / Twitter works best with 1-2. | No hashtag count was given for the platform | Platform-specific hashtag counts + automatic cap in the hashtag tool | fixed - score 8.05 -> 8.2 |

## Suggested improvements

1. Use a larger local model (e.g. Gemma 3 12B or Qwen 2.5 7B) for funnier, more natural humour and meme-style content.
2. Add a human-in-the-loop step in the UI: show 3 hooks and let the user pick before the caption is written.
3. Store platform analytics (likes/comments) back into memory so the agent learns which hooks perform best.
4. Add a fact-checking tool that compares claims in the caption with the user's README / project files (RAG).
5. Add image generation for the cover image prompt (e.g. Stable Diffusion locally) and export the post as PDF/carousel.
6. Schedule posts and track a content calendar in SQLite; deploy the FastAPI backend with Docker.

## Full outputs
Every run is saved as Markdown in `outputs/generated_posts/` or `outputs/generated_scripts/` and as JSON in `outputs/saved_results/`. Raw test logs: `outputs/logs/test_results_v1.json`, `test_results_v2.json`.