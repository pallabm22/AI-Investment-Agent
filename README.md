# 🧠 Investment Agent

An AI-powered financial assistant built with **Chainlit**, **LangChain**, and **Groq-hosted LLaMA models**, designed to deliver real-time investment insights and financial chart analysis through natural language interaction.

---

## 🚀 Features

- 💬 **Conversational Interface**: Ask questions about stock prices, company fundamentals, analyst recommendations, and historical trends.
- 📊 **Chart Analysis**: Upload financial charts (e.g., candlestick, line graphs) and describe them for AI-powered interpretation.
- 🧠 **Groq LLaMA Integration**: Uses `llama-3.3-70b-versatile` for fast, high-quality financial reasoning.
- 📈 **Markdown + Tables**: Responses are formatted for clarity using markdown and tabular data.
- 🧩 **Modular Architecture**: Built with LangChain pipelines for easy extension and customization.

---

## 🛠️ Tech Stack

- **Chainlit** – UI framework for LLM-powered apps
- **LangChain** – Prompt orchestration and agent logic
- **Groq API** – High-performance LLaMA model hosting
- **Python 3.11+** – Core language
- **Custom Tools** – YFinance + DuckDuckGo (via Phi or LangChain)

---

## 📦 Installation

```bash
git clone https://github.com/pallabm22/ai-investment-agent.git
cd ai-investment-agent
pip install -r requirements.txt
```
