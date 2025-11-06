# 🤖 Kelly — The AI Scientist Poet (Gemini 2.5)

**Kelly** is an AI-powered scientist poet chatbot built with **Streamlit** and **Google Gemini API**.  
She answers every question about AI, science, and experiments — but always in **poetic**, **analytical**, and **skeptical** verse.

---

## 🧠 Features

- 🎭 **Poetic Scientific Responses** — Every answer is in verse form.
- 🧩 **Analytical & Evidence-based** — Kelly questions assumptions and offers practical insights.
- ⚡ **Robust & Crash-Proof** — Even if Gemini refuses to answer (`finish_reason=2`), Kelly gracefully returns a *local analytical poem*.
- 🔄 **Auto Retry + Model Fallback** — Automatically retries with paraphrased prompts or alternative Gemini models.
- 🧰 **Sidebar Settings** — Switch models, clear history, or set retry attempts.
- ☁️ **Deploy-Ready** — Works locally or on Streamlit Community Cloud.

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/kelly-poet-chatbot.git
cd kelly-poet-chatbot
