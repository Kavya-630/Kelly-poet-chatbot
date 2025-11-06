# app.py — Kelly the AI Scientist Poet (robust: no finish_reason=2 crash)
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import time
import random
import html

# -------------------------------
# 0. Config / Load key
# -------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY in a .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# 1. Defaults / Persona
# -------------------------------
DEFAULT_MODEL = "models/gemini-flash-latest"   # least strict by default
FALLBACK_MODEL = "models/gemini-2.5-pro"       # optional stronger model if desired

KELLY_SYSTEM_PROMPT = """
You are Kelly — an analytical poet and scientist. Answer in the form of a short poem.
Style: skeptical, analytical, professional. Include one or two practical suggestions or steps.
Keep lines concise and clear; avoid hype and be evidence-minded.
"""

# -------------------------------
# 2. Utility: safe extraction of text from response
# -------------------------------
def extract_text_from_response(response):
    """
    Safely extract text from a genai response by iterating candidates -> content -> parts.
    Returns a string if any text found, otherwise None.
    """
    parts_out = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            txt = getattr(part, "text", None)
            if txt:
                parts_out.append(txt)
    if parts_out:
        # Join preserving existing breaks
        return "\n".join(p.strip() for p in parts_out if p.strip())
    return None

# -------------------------------
# 3. Local fallback poem generator (deterministic, helpful)
# -------------------------------
def local_poem_fallback(prompt):
    """
    Create a short, analytical poem locally if model refuses or times out.
    This is deterministic and safe — ensures user always gets a response.
    """
    # sanitize and extract a short topic phrase
    p = html.unescape(prompt).strip()
    # make a concise title-like phrase from prompt
    topic = p.replace("pipeline of", "").replace("Explain", "").replace("explain", "").strip()
    if not topic:
        topic = "this topic"
    # simple lines built from templates
    lines = [
        f"In careful lines I study {topic},",
        "A generator dreams, a critic critiques;",
        "Adversarial rhythm tunes model feats,",
        "Yet metrics warn where shortcuts meet.",
        "Practical: validate with held-out sets,",
        "Audit samples, and record your bets."
    ]
    # If prompt seems short, make the poem shorter
    if len(p.split()) <= 3:
        lines = lines[:4]
    return "\n".join(lines)

# -------------------------------
# 4. Core: attempt generation with retries + paraphrase + fallback
# -------------------------------
def kelly_reply(prompt, model_name=DEFAULT_MODEL, max_retries=3):
    """
    Attempts to get a poem from Gemini with safe extraction.
    Retries with model fallback and paraphrasing. If all fail, returns a local fallback poem.
    Returns tuple (text, source) where source is "gemini" or "fallback".
    """
    # conversation context is kept minimal to avoid over-triggering safety filters
    system = KELLY_SYSTEM_PROMPT.strip()
    base_prompt = f"{system}\n\nUser: {prompt}\nKelly:"

    attempt = 0
    # list of candidate model attempts in order (first chosen model, then flash-latest if different)
    model_attempts = [model_name]
    if "flash" not in model_name:
        model_attempts.append("models/gemini-flash-latest")
    # try paraphrases as separate attempts if model returns safety block
    paraphrases = [
        lambda q: q,
        lambda q: q + " Please explain step by step and include practical suggestions.",
        lambda q: "In scientific terms, describe the stages of: " + q,
    ]

    for mdl in model_attempts:
        for parap in paraphrases:
            attempt += 1
            if attempt > max_retries:
                break
            q = parap(prompt)
            try:
                model = genai.GenerativeModel(mdl)
                response = model.generate_content(
                    [base_prompt.replace(prompt, q)],
                    generation_config=genai.GenerationConfig(
                        temperature=0.45,
                        max_output_tokens=400,
                        top_p=0.9,
                    ),
                    # reduce over-blocking but do not bypass safety completely
                    safety_settings=[
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    ],
                )

                # robust extraction
                text = extract_text_from_response(response)
                # check finish_reason if available for debug / decision
                fr = None
                if getattr(response, "candidates", None):
                    fr = getattr(response.candidates[0], "finish_reason", None)

                if text:
                    # success
                    return text.strip(), "gemini"

                # If finish_reason explicitly indicates safety block (2), continue to next attempt
                if fr == 2:
                    # small backoff then continue
                    time.sleep(0.6)
                    continue

                # If no text but not finish_reason=2, still retry a bit
                time.sleep(0.4)
                continue

            except Exception as e:
                # transient network or API error -> wait and retry
                time.sleep(0.6)
                continue

    # If we reach here, all attempts failed — return a local fallback poem
    fallback_text = local_poem_fallback(prompt)
    return fallback_text, "fallback"

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Kelly — AI Scientist Poet (robust)", layout="centered")
st.title("🤖 Kelly — The AI Scientist Poet (robust)")
st.caption("Every answer is a poem: skeptical, analytical, and practical. (Guaranteed response)")

# Sidebar controls
with st.sidebar:
    st.header("Settings ⚙️")
    model_choice = st.selectbox(
        "Preferred model",
        options=["models/gemini-flash-latest", "models/gemini-2.5-pro", "models/gemini-2.5-flash"],
        index=0,
    )
    retries = st.slider("Max attempts", 1, 6, 3)
    if st.button("🧹 Clear conversation"):
        st.session_state.history = []

# Initialize session
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("Ask Kelly anything about AI, science, experiments, or methodology — she will answer poetically and include practical suggestions. If the API blocks the reply, Kelly will still return a helpful local poem.")

user_input = st.text_area("🧩 Your question", placeholder="e.g. Explain the pipeline of a GAN", height=120)

if st.button("💬 Ask Kelly"):
    if user_input.strip():
        st.session_state.history.append({"role": "user", "content": user_input})
        text, source = kelly_reply(user_input, model_choice, max_retries=retries)
        # annotate when fallback used
        if source == "fallback":
            text = text + "\n\n(Kelly note: the model returned no text; showing a local analytical fallback.)"
        st.session_state.history.append({"role": "assistant", "content": text})

# Display chat
if st.session_state.history:
    st.markdown("---")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍🎓 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Kelly:**\n\n{msg['content']}")

st.markdown("---")
st.markdown("💡 Tip: If you see the fallback note often, try switching to a different model in the sidebar or slightly rephrasing your question (e.g., add 'In scientific terms' or 'Step by step').")
