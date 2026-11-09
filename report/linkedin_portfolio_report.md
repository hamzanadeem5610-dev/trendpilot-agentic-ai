# TrendPilot - A Local Agentic AI Content Planner Using Ollama and Gemma 3:4B
*Hamza Nadeem - AIRI Team PITB AI Internship Task 2*

## 1. Project Overview
I built TrendPilot, a local Agentic AI assistant that runs on gemma3:4b through Ollama, so no paid API is needed. It turns any project or topic into a platform-ready content plan: a content idea, target audience, hook, caption, hashtags, a short-video script, title and thumbnail text. Instead of answering in one shot, the agent plans which tools it needs, calls them, reviews its own output against platform rules and rewrites weak parts before saving the result. The same engine also runs a StudyBuddy mode that creates study plans, notes and quizzes and tracks progress.

## 2. Tools and Technologies Used
Python, Ollama, gemma3:4b (fallbacks: Llama 3.2 3B, Qwen 2.5 3B, Phi-3 Mini), LangGraph, Streamlit, FastAPI, SQLite memory, JSON/Markdown file saving, DuckDuckGo search (optional), Google Colab (T4 GPU) and GitHub.

## 3. Agent Workflow
User gives topic -> memory recall -> agent understands the goal and audience -> planner selects tools -> tools generate content -> Content Reviewer scores it and checks platform rules -> weak parts are rewritten automatically -> File Saver stores Markdown + JSON -> final output is returned. The user can then give feedback (for example "make the hook shorter") and only the affected tools are re-run.

## 4. Tools Implemented
- **Simple Memory Tool** - Recall previous topics, platforms, styles and hooks from long-term memory (SQLite) so the new content is fresh.
- **Web Trend Research (DuckDuckGo)** - Search the web (DuckDuckGo) for fresh, real context about the topic so the content is current and factual.
- **Trend Idea Generator** - Generate 5 viral content angles for the topic and platform, score them and select the best one.
- **Caption Writer** - Write the hook, the full platform-ready post caption and a call to action in the requested tone.
- **Hashtag Generator** - Generate relevant hashtags for the platform (right count, broad + niche, no spam tags).
- **Reel Script Generator** - Create a 15-60 second short-video script: opening hook, explanation, visual suggestions and closing line.
- **Title and Thumbnail Text Generator** - Generate catchy video/post titles and short thumbnail text.
- **Post Image Prompt Generator** - Write a prompt for an AI image generator to create the post's cover image (optional extra).
- **Content Reviewer** - Review the content (hook strength, clarity, length, tone, hashtag relevance, platform readiness) and decide what to improve.
- **File Saver Tool** - Save the final result as Markdown + JSON in outputs/ (posts, scripts, full results). Mandatory last step.
- **StudyBuddy tools** - Study Planner, Notes Generator, Quiz Generator, Study Pack Reviewer and Progress Saver.

## 5. Demo and Outputs
The project has three demos: an interactive CLI (`python demo.py`), a Streamlit app (topic, platform, tone, output type, Generate, final output, Save result, live agent trace) and a FastAPI backend (`POST /generate-content`).

Example input: *I want to create a viral LinkedIn post about my YOLOv8 helmet detection project* - LinkedIn - Professional but exciting

Example hook: "I trained an AI to spot missing helmets. The hardest part wasn't the model - it was the data."

Hashtags: #YOLOv8HelmetDetection #RealTimeObjectDetection #ComputerVisionSafety #DeepLearningRobotics

Saved file: `/content/agentic_ai_task_2/outputs/generated_posts/yolov8_helmet_detection_project_linkedin_20261104_073420_post.md`

Screenshots: `screenshots/ollama_running.png`, `screenshots/demo_output.png`, `screenshots/saved_file_output.png`.

## 6. Testing
I tested the agent on 10 inputs from the brief (YOLOv8 posts, internship stories, funny Python reels, X posts, meme-style reels, portfolio posts and YouTube Shorts). The improved agent reached an average review score of 8.21/10 (baseline 7.29) and 9/10 outputs were directly usable (baseline 3/10). Weaker cases: Machine learning project.

## 7. Error Analysis
The first version broke platform rules 15 times: captions too long for X, too many and irrelevant hashtags, reel scripts longer than 60 seconds and generic cliches. I added platform-specific limits, target audience, few-shot hook examples and a banned-cliche list to the prompts, plus a reviewer loop that re-runs only the weak tool. Violations dropped to 2.

## 8. What I Learned
- How to run local LLMs with Ollama and call them from Python.
- The difference between a chatbot and an AI agent.
- How to connect tools to an LLM with a planner and a tool registry.
- Prompt design for structured JSON outputs with small models.
- How to save agent results and use SQLite memory.
- How to make an agent review and improve its own work.
- How to test, compare versions and analyse failures of an agentic workflow.

## 9. Future Improvements
- Use a larger local model (e.g. Gemma 3 12B or Qwen 2.5 7B) for funnier, more natural humour and meme-style content.
- Add a human-in-the-loop step in the UI: show 3 hooks and let the user pick before the caption is written.
- Store platform analytics (likes/comments) back into memory so the agent learns which hooks perform best.
- Add a fact-checking tool that compares claims in the caption with the user's README / project files (RAG).
- Add image generation for the cover image prompt (e.g. Stable Diffusion locally) and export the post as PDF/carousel.
- Schedule posts and track a content calendar in SQLite; deploy the FastAPI backend with Docker.

## 10. LinkedIn Post

Today, I completed AIRI Team PITB - AI Internship Task 2, where I built an end-to-end Agentic AI assistant using a local LLM through Ollama.

For this task, I developed TrendPilot, an AI agent that takes a topic and generates content ideas, LinkedIn posts, hashtags, reel scripts and saved Markdown files - running fully locally on gemma3:4b. The agent plans which tools to use, reviews its own output against platform rules and rewrites weak parts automatically. On 10 test cases, the improved version raised the average review score from 7.29 to 8.21/10. It also has a StudyBuddy mode for study plans, notes and quizzes.

This project helped me understand how agentic AI systems work beyond simple chatbot responses. I learned how to design an agent workflow, connect tools, add memory, save outputs, test different prompts, and analyze weak responses.

Tools used: Python, Ollama, gemma3:4b, LangGraph, Streamlit, FastAPI, SQLite, JSON/Markdown, and GitHub.

Key learning: Agentic AI is not only about generating text. It is about planning, using tools, storing useful information, improving outputs, and completing tasks in a structured way.

#AgenticAI #Ollama #LocalLLM #Gemma #Python #ArtificialIntelligence #AIRI #PITB #AIInternship #MachineLearning #GenerativeAI