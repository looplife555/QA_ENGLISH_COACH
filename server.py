"""
SpeakUp - AI English coach (web UI backend)
Browser mic -> faster-whisper (STT) -> LLM (local Ollama OR OpenAI/Anthropic/any API) -> Kokoro (TTS) -> browser speakers
Run:  python server.py   (opens http://localhost:8000)

LLM provider is chosen in the Setup tab (Local Ollama by default).
API keys are stored only in the browser and are forwarded to the provider you picked.
Set COACH_PROVIDER / COACH_API_KEY / COACH_BASE_URL / COACH_MODEL as environment
variables to hard-code defaults for a deployment.
"""
import io
import json
import os
import re
import tempfile
import threading
import webbrowser
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from llm_provider import (
    LLMClient,
    LLMError,
    PROVIDERS,
    resolve_settings,
    provider_info,
)

# ---------------- CONFIG ----------------
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEVICE = os.environ.get("COACH_DEVICE", "cpu")
COMPUTE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_DEFAULT = os.environ.get("COACH_WHISPER", "small.en" if DEVICE == "cuda" else "base.en")
PORT = int(os.environ.get("COACH_PORT", "8000"))
DEFAULT_VOICE = "af_heart"

CATALOG = [  # sizes are approximate downloads
    {"id": "qwen3:1.7b", "name": "Qwen 3 1.7B", "size": "1.4 GB", "note": "Fast and still reliable. Great for live chat", "rec": True},
    {"id": "qwen3:0.6b", "name": "Qwen 3 0.6B", "size": "0.5 GB", "note": "Fastest, but corrections are unreliable", "rec": False},
    {"id": "qwen3:4b", "name": "Qwen 3 4B", "size": "2.6 GB", "note": "Best balance for a laptop CPU", "rec": True},
    {"id": "qwen2.5:3b", "name": "Qwen 2.5 3B", "size": "1.9 GB", "note": "Fast, solid grammar feedback", "rec": True},
    {"id": "qwen2.5:1.5b", "name": "Qwen 2.5 1.5B", "size": "0.9 GB", "note": "Fastest, simpler feedback", "rec": False},
    {"id": "llama3.2:3b", "name": "Llama 3.2 3B", "size": "2.0 GB", "note": "Natural, friendly tone", "rec": False},
    {"id": "phi3:mini", "name": "Phi-3 Mini 3.8B", "size": "2.3 GB", "note": "Strong reasoning, a bit slower", "rec": False},
    {"id": "qwen3:8b", "name": "Qwen 3 8B", "size": "5.2 GB", "note": "Best feedback, slow on CPU", "rec": False},
]
WHISPER_OPTIONS = [
    {"id": "tiny.en", "label": "Tiny", "note": "Fastest, least accurate (~75 MB)"},
    {"id": "base.en", "label": "Base", "note": "Good speed and accuracy (~145 MB)"},
    {"id": "small.en", "label": "Small", "note": "Most accurate, slower on CPU (~485 MB)"},
]


def load_catalog():
    """Built-in list + optional models.json next to this file (edit it to add your own models; no restart needed)."""
    items = {c["id"]: dict(c) for c in CATALOG}
    try:
        extra = json.loads(Path(__file__).with_name("models.json").read_text(encoding="utf-8"))
        for e in extra if isinstance(extra, list) else []:
            mid = str(e.get("id", "")).strip()
            if re.match(r"^[\w.:/-]{1,120}$", mid):
                items[mid] = {
                    "id": mid,
                    "name": str(e.get("name") or mid),
                    "size": str(e.get("size") or ""),
                    "note": str(e.get("note") or "Custom model"),
                    "rec": bool(e.get("rec", False)),
                }
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"models.json ignored: {ex}")
    return list(items.values())


VOICES = [
    {"id": "af_heart", "label": "Heart", "accent": "American", "gender": "Female"},
    {"id": "af_bella", "label": "Bella", "accent": "American", "gender": "Female"},
    {"id": "af_nicole", "label": "Nicole", "accent": "American", "gender": "Female"},
    {"id": "af_sky", "label": "Sky", "accent": "American", "gender": "Female"},
    {"id": "am_michael", "label": "Michael", "accent": "American", "gender": "Male"},
    {"id": "am_adam", "label": "Adam", "accent": "American", "gender": "Male"},
    {"id": "bf_emma", "label": "Emma", "accent": "British", "gender": "Female"},
    {"id": "bf_isabella", "label": "Isabella", "accent": "British", "gender": "Female"},
    {"id": "bm_george", "label": "George", "accent": "British", "gender": "Male"},
    {"id": "bm_lewis", "label": "Lewis", "accent": "British", "gender": "Male"},
]
VOICE_IDS = {v["id"] for v in VOICES}

ABOUT = ("The learner is Rohit, a software QA professional with 5-6 years of experience, mainly manual testing, "
         "in BFSI (wealth management). He tests financial calculations, validates cash-flow records, checks "
         "portfolio performance, and investigates missing cash-flow records and incorrect performance calculations. "
         "This is communication practice only: never pretend to verify real financial data and never invent facts about his work.")

RULES = """
Speaking rules:
- Reply in 1-3 short, natural sentences, like real speech. No lists, markdown, emojis or stage directions.
- Use clear, simple, professional English. Be warm and encouraging.
- After your spoken reply, add a line containing only ### and then at most 3 short coaching notes, each on its own line
  starting with "- " (grammar, better phrasing, filler words, clarity), based on Rohit's last message.
  Gently show the corrected version of one mistake. If his English was fine, write only: Looks good.
  The notes are shown on screen only, never spoken.
Example:
That sounds like a tough bug. What did you check first?
###
- Say "I have been working" instead of "I am working since".
"""

RULES_LIVE = """
Speaking rules:
- Reply in 1-3 short, natural sentences, like real speech. No lists, markdown, emojis or stage directions.
- Use clear, simple, professional English. Be warm and encouraging.
- Do not add coaching notes or corrections. Just keep the conversation going naturally.
"""

IMPROVE_PROMPT = """You are an English communication coach. Your learner is Rohit, a software QA professional (5-6 years,
manual testing, BFSI / wealth management). He drafted an answer for an interview or workplace conversation.

Task:
1. Rewrite his draft into a natural, confident, spoken answer in 2-4 short sentences (spoken English, not an essay).
2. Then add a line containing only ### and up to 3 short bullets: what changed and why, plus one phrase worth memorizing.
- Never invent facts about his work.
- The part after ### is shown on screen only, never spoken.
Example:
I found the defect in the cash-flow feed and raised it with the team.
###
- "Raised it" -> "Escalated it" (stronger QA verb).
- Add a number: how many records were affected.
"""

SCORECARD_PROMPT = """You are an expert English communication coach. Your learner is Rohit, a software QA professional
(5-6 years, manual testing, BFSI / wealth management).

Review the transcript below for the scenario "{title}" with goals: {goals}.
You only have speech-to-text text, so do NOT comment on pronunciation or accent.
Rate each area from 1 to 10 and give one short, kind comment per area. Quote Rohit's exact words where useful.
Be specific, encouraging and practical. Plain text only, exactly these lines:

Fluency: X/10 - ...
Grammar: X/10 - ...
Vocabulary: X/10 - ...
Structure: X/10 - ...
Confidence: X/10 - ...
Overall: X/10 - ...
One thing to try next: ...

Stats: {stats}"""

REPORT_PROMPT = """You are an expert English communication coach reviewing a full practice session transcript.
{about}
Scenario practised: {title}. Goals: {goals}.
You only have text transcripts from speech-to-text, so do NOT comment on pronunciation or accent. Judge grammar, word choice,
sentence structure, clarity, fluency signals (fillers, repetition, very short answers) and the scenario goals.
Be specific and kind. Quote Rohit's exact words. Never invent things he did not say.
Write in plain text using exactly these headings, each followed by short bullet points starting with "- ":
## Overall
## What you did well
## Mistakes to fix   (for each: What you said -> Better version -> Why)
## Vocabulary upgrades   (plain words he used -> stronger work-appropriate alternatives, with one example sentence)
## Fluency and structure
## Scenario feedback
## Your 3 goals for next session"""

SCENARIOS = [
    {"id": "daily-standup", "title": "Daily Standup", "tagline": "Yesterday, today, blockers in 60 seconds", "level": "Beginner",
     "minutes": "5 min", "icon": "users",
     "description": "Practise the classic three-point stand-up with short, clear sentences. A perfect warm-up for every workday.",
     "goals": ["Use past, present and future tenses cleanly", "Cut filler words (um, actually, basically)", "Finish your update in under 60 seconds"],
     "starter": "Good morning Rohit! Let's do your stand-up. What did you complete yesterday, what are you testing today, and is anything blocking you?",
     "followUps": ["Give me a stand-up topic", "How do I say my build is blocked?", "Help me describe yesterday's testing"],
     "prompt": "You are a friendly Scrum Master running a daily stand-up. Keep it concise, ask one follow-up at a time, and encourage short stand-up language."},
    {"id": "bug-report", "title": "Explain a Bug", "tagline": "Missing cash-flow, wrong performance", "level": "Intermediate",
     "minutes": "8 min", "icon": "bug",
     "description": "Your core strength: investigating missing cash-flow records and incorrect performance. Learn to explain defects crisply to developers.",
     "goals": ["Structure it: situation, action, result", "Use precise QA verbs: reproduced, validated, isolated", "Quantify impact: accounts, amounts, dates"],
     "starter": "Let's role-play. I'm the developer. You found missing cash-flow records for three portfolios after last night's feed. Explain the bug: severity, steps, expected versus actual.",
     "followUps": ["How do I say severity is high?", "Help me use 'expected versus actual'", "Give me polite ways to push back"],
     "prompt": "You are a developer. Rohit explains defects about missing cash flows and wrong performance numbers. Ask the clarifying questions a developer would ask (environment, steps, data, frequency), one at a time."},
    {"id": "client-status", "title": "Client Status Call", "tagline": "Polite, confident, no jargon overload", "level": "Advanced",
     "minutes": "10 min", "icon": "phone",
     "description": "Practise weekly client calls: validation status, timelines and risks, in a polite and confident tone.",
     "goals": ["Soften language: 'we noticed' instead of 'you did wrong'", "Give status with dates and next steps", "Handle tough questions calmly"],
     "starter": "Hello Rohit, I'm from the wealth desk. Our advisors say portfolio performance looks off for March. Can you walk me through what your team validated?",
     "followUps": ["How do I give a delay politely?", "Help me explain this without jargon", "Give me phrases to end the call"],
     "prompt": "You are a non-technical wealth-management client. Ask natural business questions about status, impact and timelines. Be polite but a little demanding, one question at a time."},
    {"id": "interview-qa", "title": "QA Interview", "tagline": "Manual testing + BFSI questions", "level": "Intermediate",
     "minutes": "15 min", "icon": "brief", "askRole": True,
     "description": "A realistic interviewer asks one question at a time, from manual testing to wealth-management scenarios, and checks how well your answers are structured.",
     "goals": ["Introduce yourself in 60-90 seconds", "Tell project stories with STAR (situation, task, action, result)", "Sound confident about domain scenarios"],
     "starter": "Hello Rohit, thanks for joining. Let's begin. Could you please introduce yourself?",
     "followUps": ["Give me a tricky question", "Help me answer 'tell me about a missed defect'", "How do I explain my weakness?"],
     "prompt": "You are a professional interviewer hiring for a {role} position. Run a realistic interview. Ask exactly one question at a time. Mix: self-introduction, past projects, manual testing techniques, test case design, defect handling, BFSI and wealth-management scenarios, communication and behavioural questions. After each answer, react in one short sentence, then ask the next question. In the coaching notes also check answer structure (for example STAR) and, where useful, give a stronger way to phrase one key sentence."},
    {"id": "portfolio-deepdive", "title": "Domain Talk: Portfolios", "tagline": "Cash flows, returns, reconciliation", "level": "Advanced",
     "minutes": "10 min", "icon": "chart",
     "description": "Explain wealth-management concepts such as cash flows, performance and reconciliation in simple, accurate English.",
     "goals": ["Explain domain terms to a non-expert", "Use reconcile, discrepancy and validate naturally", "Give one concrete example per answer"],
     "starter": "Hi Rohit. Imagine I'm a new joiner. In simple words, how do you check whether a portfolio's performance number is correct?",
     "followUps": ["Help me explain cash flow simply", "What is a good analogy for reconciliation?", "Give me a harder question"],
     "prompt": "You are a curious new joiner who knows little about portfolio performance and cash flows. Ask simple follow-up questions that make Rohit explain clearly. Never state financial facts as certain; stay in the role of a learner."},
    {"id": "incident-triage", "title": "Defect Triage Meeting", "tagline": "Severity, priority, assertive and polite", "level": "Advanced",
     "minutes": "10 min", "icon": "alert",
     "description": "Defend your severity rating, handle pushback and agree next steps in a triage meeting.",
     "goals": ["Be assertive but respectful", "Use evidence: 'The data shows...'", "Agree owners and dates"],
     "starter": "Welcome to triage, Rohit. Developers think the missing cash-flow defect is Medium severity. Why do you rate it High?",
     "followUps": ["Give me assertive phrases", "How do I disagree politely?", "Help me summarise next steps"],
     "prompt": "You are a project manager chairing a defect triage meeting with developers who push back on severity. Challenge Rohit's reasoning politely, one point at a time."},
    {"id": "small-talk", "title": "Office Small Talk", "tagline": "Chai breaks, weekends, lunch", "level": "Beginner",
     "minutes": "5 min", "icon": "cup",
     "description": "Build easy daily communication with colleagues: weekend plans, lunch, travel, cricket, anything casual.",
     "goals": ["Keep a casual conversation going", "Ask questions back", "Use natural everyday phrases"],
     "starter": "Hey Rohit! Good to see you. How was your weekend?",
     "followUps": ["Give me a casual topic", "How do I politely end a chat?", "Teach me a useful idiom"],
     "prompt": "You are a friendly colleague having casual small talk during a break. Share a tiny bit about yourself and ask Rohit one question at a time. Occasionally teach one useful idiom."},
    {"id": "free-talk", "title": "Free Conversation", "tagline": "Any topic you like", "level": "Beginner",
     "minutes": "5+ min", "icon": "chat",
     "description": "Talk about anything. Great for building fluency and confidence without pressure.",
     "goals": ["Keep speaking without stopping", "Add 'because...' and an example", "Aim for 25-40 words per turn"],
     "starter": "Hi Rohit! Let's practise. What would you like to talk about today?",
     "followUps": ["Give me a topic", "Ask me something interesting", "Teach me a new word"],
     "prompt": "You are a warm English speaking coach having a free conversation. End most turns with a simple follow-up question. Occasionally teach one useful word."},
]
SCEN = {s["id"]: s for s in SCENARIOS}

# ---------------- MODELS ----------------
from faster_whisper import WhisperModel  # noqa: E402
from kokoro import KPipeline  # noqa: E402

lock = threading.Lock()
whispers: dict = {}
pipelines: dict = {}


def get_whisper(size: str):
    if size not in {w["id"] for w in WHISPER_OPTIONS}:
        size = WHISPER_DEFAULT
    if size not in whispers:
        print(f"Loading Whisper '{size}' (downloads on first use)...")
        whispers[size] = WhisperModel(size, device=DEVICE, compute_type=COMPUTE)
    return whispers[size]


def get_pipeline(voice: str):
    lang = "b" if voice.startswith("b") else "a"
    if lang not in pipelines:
        print(f"Loading Kokoro ({'British' if lang == 'b' else 'American'})...")
        pipelines[lang] = KPipeline(lang_code=lang)
    return pipelines[lang]


print(f"Starting up ({DEVICE}). First run downloads the voice models, please wait...")
get_whisper(WHISPER_DEFAULT)
get_pipeline(DEFAULT_VOICE)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    provider, api_key, base_url, model = resolve_settings()
    info = {
        "provider": provider,
        "providerInfo": provider_info(provider, api_key, base_url, model),
        "ollama": False,
        "models": [],
        "defaultModel": model,
        "catalog": load_catalog(),
        "device": DEVICE,
        "whisper": {"default": WHISPER_DEFAULT, "options": WHISPER_OPTIONS, "loaded": list(whispers)},
        "voices": VOICES,
        "defaultVoice": DEFAULT_VOICE,
        "scenarios": [{k: v for k, v in s.items() if k != "prompt"} for s in SCENARIOS],
    }
    if provider == "ollama":
        try:
            r = requests.get(f"{OLLAMA}/api/tags", timeout=2.5)
            info["models"] = [{"name": m.get("name", ""), "size": m.get("size", 0)} for m in r.json().get("models", [])]
            info["ollama"] = True
        except Exception:
            pass
    else:
        # API providers: reachability can't be proven without a key; mark usable when a key is configured.
        info["ollama"] = bool(api_key)
    return info


# Chat / improve / scorecard / report share provider settings, parsed into this model.
class AISettings(BaseModel):
    provider: str | None = None
    apiKey: str | None = None
    baseUrl: str | None = None
    model: str | None = None


@app.post("/api/probe")
def probe(settings: AISettings = AISettings()):
    provider, api_key, base_url, model = resolve_settings(settings)
    client = LLMClient(provider, api_key, base_url, model)
    try:
        reply = client.chat(
            "You are a helpful assistant. Reply with exactly: PONG",
            [{"role": "user", "content": "Say PONG"}],
            temperature=0,
            max_tokens=10,
        )
    except LLMError as e:
        return JSONResponse({"ok": False, "error": str(e), "code": getattr(e, "code", None)}, status_code=200)
    return {"ok": True, "provider": provider, "model": model, "reply": reply.strip()}


class ChatReq(AISettings):
    scenario: str = "daily-standup"
    role: str = ""
    messages: list[dict]
    notes: bool = True


@app.post("/api/chat")
def chat(req: ChatReq):
    provider, api_key, base_url, model = resolve_settings(req)
    sc = SCEN.get(req.scenario, SCENARIOS[0])
    role = req.role.strip() or "Senior QA Engineer in BFSI / wealth management"
    prompt = f"{sc['prompt'].replace('{role}', role)} {ABOUT} {RULES if req.notes else RULES_LIVE}"
    msgs = [{"role": m["role"], "content": str(m["content"])[:4000]} for m in req.messages[-24:]
            if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    client = LLMClient(provider, api_key, base_url, model)
    try:
        full = client.chat(prompt, msgs, temperature=0.6, max_tokens=260 if req.notes else 120)
    except LLMError as e:
        return JSONResponse({"error": str(e), "code": getattr(e, "code", None)},
                            status_code=404 if getattr(e, "code", None) == "MODEL_MISSING" else 502)
    if not full:
        return JSONResponse({"error": "The model returned an empty reply. Try again."}, status_code=502)
    reply, _, notes = full.partition("###")
    return {"reply": reply.strip(), "notes": notes.strip() if req.notes else "", "stats": {}}


@app.post("/api/improve")
def improve(req: ChatReq):
    """Answer Coach: polish a drafted answer into a stronger spoken version."""
    provider, api_key, base_url, model = resolve_settings(req)
    sc = SCEN.get(req.scenario, SCENARIOS[0])
    role = req.role.strip() or "Senior QA Engineer in BFSI / wealth management"
    system = f"{IMPROVE_PROMPT}\nScenario: {sc['title']} ({sc['tagline']}). Role: {role}."
    msgs = [{"role": m["role"], "content": str(m["content"])[:4000]} for m in req.messages[-12:]
            if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    client = LLMClient(provider, api_key, base_url, model)
    try:
        full = client.chat(system, msgs, temperature=0.5, max_tokens=500)
    except LLMError as e:
        return JSONResponse({"error": str(e), "code": getattr(e, "code", None)}, status_code=502)
    improved, _, notes = full.partition("###")
    return {"improved": improved.strip(), "notes": notes.strip()}


class ReportReq(AISettings):
    scenario: str = "daily-standup"
    role: str = ""
    messages: list[dict]
    stats: dict = {}
    deep: bool = True


@app.post("/api/report")
def report(req: ReportReq):
    provider, api_key, base_url, model = resolve_settings(req)
    sc = SCEN.get(req.scenario, SCENARIOS[0])
    lines = []
    for m in req.messages[-60:]:
        txt = str(m.get("content", "")).strip()
        if txt and m.get("role") in ("user", "assistant"):
            lines.append(f"{'Rohit' if m['role'] == 'user' else 'Coach'}: {txt[:1500]}")
    if not any(l.startswith("Rohit:") for l in lines):
        return JSONResponse({"error": "Say something first, then I can write your report."}, status_code=400)
    st_ = req.stats or {}
    stats = (f"Session stats measured by the app: {st_.get('turns', '?')} turns, {st_.get('words', '?')} words, "
             f"{st_.get('fillers', '?')} filler words, average speaking speed {st_.get('wpm') or 'n/a'} words per minute.")
    system = REPORT_PROMPT.format(about=ABOUT, title=sc["title"], goals="; ".join(sc["goals"]))
    client = LLMClient(provider, api_key, base_url, model)

    def gen():
        yield (json.dumps({"k": "start", "think": False}) + "\n").encode()
        try:
            for chunk in client.stream_chat(system,
                                            [{"role": "user", "content": "Transcript:\n" + "\n".join(lines) + "\n\n" + stats + "\nWrite the report now."}],
                                            temperature=0.4, max_tokens=2200):
                yield (json.dumps({"k": "content", "t": chunk}) + "\n").encode()
        except LLMError as e:
            yield (json.dumps({"error": str(e), "code": getattr(e, "code", None)}) + "\n").encode()
        except Exception as e:
            yield (json.dumps({"error": f"Report failed: {e}"}) + "\n").encode()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/scorecard")
def scorecard(req: ReportReq):
    """Post-session scores across fluency, grammar, vocabulary, structure, confidence."""
    provider, api_key, base_url, model = resolve_settings(req)
    sc = SCEN.get(req.scenario, SCENARIOS[0])
    lines = []
    for m in req.messages[-60:]:
        txt = str(m.get("content", "")).strip()
        if txt and m.get("role") in ("user", "assistant"):
            lines.append(f"{'Rohit' if m['role'] == 'user' else 'Coach'}: {txt[:1500]}")
    if not any(l.startswith("Rohit:") for l in lines):
        return JSONResponse({"error": "Say something first, then I can score your session."}, status_code=400)
    st_ = req.stats or {}
    stats = (f"Session stats measured by the app: {st_.get('turns', '?')} turns, {st_.get('words', '?')} words, "
             f"{st_.get('fillers', '?')} filler words, average speaking speed {st_.get('wpm') or 'n/a'} words per minute.")
    system = SCORECARD_PROMPT.format(title=sc["title"], goals="; ".join(sc["goals"]), stats=stats)
    client = LLMClient(provider, api_key, base_url, model)

    def gen():
        yield (json.dumps({"k": "start", "think": False}) + "\n").encode()
        try:
            for chunk in client.stream_chat(system,
                                            [{"role": "user", "content": "Transcript:\n" + "\n".join(lines) + "\nWrite the scorecard now."}],
                                            temperature=0.4, max_tokens=900):
                yield (json.dumps({"k": "content", "t": chunk}) + "\n").encode()
        except LLMError as e:
            yield (json.dumps({"error": str(e), "code": getattr(e, "code", None)}) + "\n").encode()
        except Exception as e:
            yield (json.dumps({"error": f"Scorecard failed: {e}"}) + "\n").encode()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class PullReq(BaseModel):
    model: str


@app.post("/api/pull")
def pull(req: PullReq):
    provider, api_key, base_url, model = resolve_settings(req)
    if provider != "ollama":
        return JSONResponse({"error": "Downloads only apply to local Ollama. API providers have nothing to install."}, status_code=400)
    name = req.model.strip()
    if not re.match(r"^[\w.:/-]{1,120}$", name):
        return JSONResponse({"error": "Invalid model name."}, status_code=400)

    def gen():
        try:
            with requests.post(f"{OLLAMA}/api/pull", json={"model": name, "stream": True}, stream=True, timeout=None) as r:
                for line in r.iter_lines():
                    if line:
                        yield line + b"\n"
        except Exception as e:
            yield (json.dumps({"error": f"Can't reach Ollama: {e}"}) + "\n").encode()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


class TTSReq(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0


@app.post("/api/tts")
def speak(req: TTSReq):
    text = req.text.strip()[:1200]
    if not text:
        return JSONResponse({"error": "No text."}, status_code=400)
    voice = req.voice if req.voice in VOICE_IDS else DEFAULT_VOICE
    speed = min(1.3, max(0.7, req.speed))
    with lock:
        pipe = get_pipeline(voice)
        chunks = []
        for _, _, audio in pipe(text, voice=voice, speed=speed):
            chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"\nReady -> {url}   (Ctrl+C to stop)\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
