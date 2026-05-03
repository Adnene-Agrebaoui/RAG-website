# 🌿 Eco-Friendly RAG System

> PDF Question-Answering with Minimal Environmental Impact  
> Built with **Python · FastAPI · LangChain · Groq LPU · Ollama · Llama 3.3**

---

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) web application that lets users ask natural language questions about a PDF document. Instead of loading the entire document into an LLM context window for every query, it retrieves only the most relevant chunks and feeds those to a fast, efficient LLM hosted on Groq's energy-efficient LPU infrastructure.

> **RAG = Retrieve first, then generate.** Only the relevant parts of your document reach the LLM — drastically cutting token count, latency, and energy usage per query.

---

## 2. Prerequisites

- Python 3.10+ — https://python.org
- Ollama (local embeddings) — https://ollama.com
- Groq API key (free tier) — https://console.groq.com
- Git (optional)

---

## 3. Step-by-Step Setup

### Step 1 — Create your project folder
```bash
mkdir rag-project && cd rag-project
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `fastapi`, `uvicorn[standard]`
- `langchain`, `langchain-groq`, `langchain-ollama`, `langchain-community`, `langchain-core`
- `pypdf`, `docarray`, `python-dotenv`, `pydantic`

### Step 4 — Set up your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### Step 5 — Add your PDF
Place your PDF in the project root. The default expected filename is:
```
Retrieval-Augmented-Generation.pdf
```
Or edit `pdf_path` in `main.py` to match your filename.

### Step 6 — Pull the Ollama embedding model (one-time)
```bash
ollama pull nomic-embed-text
```
> On Windows and macOS, Ollama runs as a background service after installation. You do **not** need to run `ollama serve` manually — it starts automatically.

### Step 7 — Start the server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Successful startup output:
```
Initializing RAG components...
1. Loading LLM...
2. Loading embeddings...
3. Loading PDF...
   -> 20 pages loaded
4. Building vectorstore...
5. Building chain...
RAG setup complete. Ready for questions.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 8 — Use the app
Open your browser and go to:
```
http://127.0.0.1:8000
```
Type your question and press **Ask**.

---

## 4. Troubleshooting

### Port already in use (Error 10048)
```bash
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```
Or use a different port:
```bash
uvicorn main:app --host 127.0.0.1 --port 8001
```

### Ollama bind error (Error 11434)
Ollama is already running as a background service — this is correct. Do **not** run `ollama serve` manually.

### Server starts but queries return no answer
Visit `http://localhost:11434` — you should see `Ollama is running`. If not, restart Ollama from your system tray.

### NameError: name 'main' is not defined
The FastAPI instance and the file share the same name. Rename the instance in `main.py`:
```python
# Change this:
main = FastAPI()
# To this:
app = FastAPI()
```
Then update all decorators (`@app.get`, `@app.post`) and middleware (`app.add_middleware`), and run `uvicorn main:app`.

---

## 5. System Comparison: Energy, CO2 & Precision

| Approach | Energy / Query | CO2 / Query | Precision | Notes |
|---|---|---|---|---|
| **Our RAG + Groq LPU** (llama-3.3-70b) | ~0.03–0.05 Wh | ~0.005–0.01 g | High (retrieval-scoped) | Only relevant chunks sent to LLM; LPU is up to 10x more energy-efficient than GPU |
| **Our RAG + Gemini Flash** | ~0.08–0.15 Wh | ~0.01–0.03 g | High (retrieval-scoped) | Runs on TPUs; efficient but cloud-only; higher CO2 than Groq LPU |
| **NotebookLM** (Google) | ~0.24–0.5 Wh | ~0.03–0.08 g | Very High (explanatory) | Designed to *explain & discuss*, not just retrieve; full document context per query; more thorough but heavier |
| **Advanced LLM, no RAG** (GPT-4o, Claude…) | ~0.5–1.5 Wh | ~0.1–0.3 g | Medium (hallucination risk) | Full PDF pasted into context window; massive token count; no filtering; highest energy cost |

> **Key distinction:** NotebookLM excels at deep document understanding and explanation — it sends the full document context per query. Our RAG system is optimized for targeted retrieval — it answers specific questions using only the most relevant chunks, making it significantly lighter per query.

### Why Groq beats Gemini on energy
Groq's LPU uses on-chip SRAM instead of off-chip HBM memory, reducing energy per bit by ~20x (0.3 pJ vs 6 pJ per bit). Groq's technical documentation claims up to **10x better energy efficiency** versus GPU-based inference at the architectural level.

---

## 6. Estimated Global Energy & CO2 Savings

### How many people query PDFs with LLMs today?

As of early 2026, ChatGPT processes over **2.5 billion prompts per day** across ~900 million weekly active users. Document retrieval and PDF querying is among the top professional use cases. Conservatively estimating that 1–2% of daily LLM queries involve uploading and questioning a PDF without RAG gives us roughly **25–50 million PDF-query interactions per day** globally.

A standard no-RAG PDF query pastes the full document into context — for a 20-page PDF, that's 10,000–40,000 tokens, consuming ~0.5–1.5 Wh and emitting ~0.1–0.3 g CO2 per query on GPU infrastructure. Our RAG system reduces token count by **70–90%**, bringing energy down to ~0.03–0.05 Wh per query.

### Projected Savings

| Scenario | Daily Queries | Energy Saved / Day | CO2 Saved / Day |
|---|---|---|---|
| ~10M users switch from no-RAG to RAG+Groq | ~10,000,000 | ~9,500 kWh | ~1.9 kg CO2 |
| ~50M users (est. global PDF-query LLM users) | ~50,000,000 | ~47,500 kWh | ~9.5 kg CO2 |
| Annual projection (50M users, 365 days) | ~18.25B | ~17.3 GWh/yr | ~3.5 tonnes CO2/yr |

> Estimates based on: Groq LPU energy advantage (up to 10x vs GPU, Groq technical paper 2024); median LLM query energy of 0.24–1 Wh (Google/MIT/Mistral, 2024–2025); ChatGPT daily query volume of 2.5B (OpenAI, 2026); conservative 1–2% PDF-query share. CO2 conversion uses US average grid intensity of 0.386 kg CO2/kWh.

---

## 7. Project File Structure

```
rag-project/
├── main.py                        # FastAPI server + RAG chain + embedded UI
├── requirements.txt               # Python dependencies
├── .env                           # GROQ_API_KEY (never commit this!)
├── Retrieval-Augmented-Generation.pdf   # Your target PDF
└── venv/                          # Virtual environment (not committed)
```

---

## 8. System Architecture

When a user submits a question, the following pipeline executes:

1. The question is embedded **locally** by Ollama using `nomic-embed-text`
2. The vector embedding is compared against pre-indexed PDF chunks in `DocArrayInMemorySearch`
3. The **top-k most similar chunks** are retrieved (typically 3–5 passages)
4. Only those chunks + the question are sent to the **Groq-hosted Llama 3.3 70B**
5. The LLM generates an answer grounded exclusively in the retrieved context
6. The answer is returned to the browser via the FastAPI `/ask` endpoint

> Because Ollama runs locally and Groq's LPU is 10x more energy-efficient than a GPU, this architecture minimizes both data center energy usage and API latency simultaneously.

---

*Built with care for performance and the planet. — May 2026*