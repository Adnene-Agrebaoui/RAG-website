import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from operator import itemgetter

# LangChain & Model Imports
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch

# 1. INITIALIZATION
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DATA MODELS
class QuestionRequest(BaseModel):
    question: str

# 3. RAG SETUP
print("Initializing RAG components...")
my_key = os.environ.get("GROQ_API_KEY")

print("1. Loading LLM...")
llm = ChatGroq(groq_api_key=my_key, model_name="llama-3.3-70b-versatile")

print("2. Loading embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("3. Loading PDF...")
loader = PyPDFLoader("Retrieval-Augmented-Generation.pdf")
pages = loader.load_and_split()
print(f"   → {len(pages)} pages loaded")

print("4. Building vectorstore...")
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("5. Building chain...")
template = """Answer the question based only on context: {context}
Question: {question}"""
prompt = PromptTemplate.from_template(template)
chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt | llm | StrOutputParser()
)
print("RAG setup complete. Ready for questions.")
# 4. SERVE THE UI
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Ask the PDF</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0d0d0d;
      --surface: #161616;
      --border: #2a2a2a;
      --accent: #c8a96e;
      --accent-dim: rgba(200, 169, 110, 0.12);
      --text: #e8e2d9;
      --text-muted: #6b6560;
      --font-display: 'Playfair Display', Georgia, serif;
      --font-mono: 'DM Mono', 'Courier New', monospace;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-mono);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 60px 20px 40px;
    }

    header {
      text-align: center;
      margin-bottom: 52px;
    }

    header .eyebrow {
      font-family: var(--font-mono);
      font-size: 10px;
      letter-spacing: 0.3em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 14px;
    }

    header h1 {
      font-family: var(--font-display);
      font-size: clamp(2rem, 5vw, 3.2rem);
      font-weight: 700;
      letter-spacing: -0.02em;
      line-height: 1.1;
      color: var(--text);
    }

    header h1 em {
      font-style: italic;
      color: var(--accent);
    }

    .card {
      width: 100%;
      max-width: 680px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      overflow: hidden;
    }

    .input-area {
      padding: 28px 28px 0;
    }

    .input-label {
      font-size: 10px;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 10px;
      display: block;
    }

    .input-row {
      display: flex;
      gap: 10px;
      align-items: stretch;
    }

    #question {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 13px 16px;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s;
    }

    #question::placeholder { color: var(--text-muted); }
    #question:focus { border-color: var(--accent); }

    #askBtn {
      background: var(--accent);
      color: #0d0d0d;
      border: none;
      border-radius: 3px;
      padding: 0 22px;
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
      white-space: nowrap;
    }

    #askBtn:hover { opacity: 0.85; }
    #askBtn:active { transform: scale(0.98); }
    #askBtn:disabled { opacity: 0.4; cursor: not-allowed; }

    .divider {
      height: 1px;
      background: var(--border);
      margin: 28px 0 0;
    }

    .response-area {
      padding: 24px 28px 28px;
      min-height: 140px;
    }

    .response-label {
      font-size: 10px;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--text-muted);
      transition: background 0.3s;
    }
    .status-dot.active { background: var(--accent); }
    .status-dot.thinking {
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.2; }
    }

    #response {
      font-size: 15px;
      line-height: 1.8;
      color: var(--text);
      white-space: pre-wrap;
    }

    #response.placeholder { color: var(--text-muted); font-style: italic; }
    #response.error { color: #e07070; }

    footer {
      margin-top: 40px;
      font-size: 11px;
      color: var(--text-muted);
      letter-spacing: 0.1em;
    }
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">RAG · Retrieval-Augmented Generation</p>
    <h1>Ask the <em>Document</em></h1>
  </header>

  <div class="card">
    <div class="input-area">
      <span class="input-label">Your question</span>
      <div class="input-row">
        <input type="text" id="question" placeholder="What would you like to know?" autocomplete="off"/>
        <button id="askBtn" onclick="askQuestion()">Ask</button>
      </div>
    </div>
    <div class="divider"></div>
    <div class="response-area">
      <div class="response-label">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusLabel">Awaiting query</span>
      </div>
      <div id="response" class="placeholder">The answer will appear here once you ask a question.</div>
    </div>
  </div>

  <footer>Powered by Groq · LangChain · Ollama</footer>

  <script>
    const questionInput = document.getElementById('question');
    const responseDiv = document.getElementById('response');
    const btn = document.getElementById('askBtn');
    const dot = document.getElementById('statusDot');
    const statusLabel = document.getElementById('statusLabel');

    function setStatus(state) {
      dot.className = 'status-dot';
      if (state === 'thinking') {
        dot.classList.add('thinking');
        statusLabel.textContent = 'Thinking…';
      } else if (state === 'done') {
        dot.classList.add('active');
        statusLabel.textContent = 'Answer ready';
      } else if (state === 'error') {
        statusLabel.textContent = 'Error';
      } else {
        statusLabel.textContent = 'Awaiting query';
      }
    }

    questionInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') askQuestion();
    });

    async function askQuestion() {
      const q = questionInput.value.trim();
      if (!q) return;

      btn.disabled = true;
      responseDiv.className = '';
      responseDiv.textContent = '';
      setStatus('thinking');

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        if (data.answer) {
          responseDiv.textContent = data.answer;
          setStatus('done');
        } else {
          responseDiv.className = 'error';
          responseDiv.textContent = data.detail || 'Unexpected response from server.';
          setStatus('error');
        }
      } catch (err) {
        responseDiv.className = 'error';
        responseDiv.textContent = 'Could not reach the server. Is uvicorn running?';
        setStatus('error');
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTML_PAGE

# 5. API ENDPOINT
@app.post("/ask")
async def ask_rag(request: QuestionRequest):
    try:
        response = chain.invoke({"question": request.question})
        return {"answer": response}
    except Exception as e:
        print(f"Chain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 6. EXECUTION BLOCK
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    print("Open your browser at: http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)