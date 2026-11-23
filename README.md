# SmartSupport — AI Customer Support Chatbot with Memory

An AI support assistant for **The Chronos Collective**, a (fictional) online watch store.
It answers customer questions **only from the company knowledge base** using RAG, refuses
off-topic questions with guardrails, and remembers each user's conversation in MongoDB.

**Tech stack:** Python · LangChain · FastAPI · FAISS · MongoDB · Gemini API (OpenAI supported)

---

## Features

- **RAG (Retrieval-Augmented Generation):** the knowledge base is split into chunks, embedded, and stored in a FAISS vector index. For every question, the most relevant chunks are retrieved and given to the LLM as context.
- **Prompt templates:** a LangChain `ChatPromptTemplate` with strict rules, the retrieved context, the chat history and the question.
- **Guardrails for out-of-scope questions** (3 layers):
  1. Greetings and thanks get fixed replies, so no LLM call is needed.
  2. **Relevance threshold:** if the best FAISS match has a cosine similarity below `0.62`, the question is treated as out of scope and refused *before* calling the LLM. This saves cost and prevents hallucination.
  3. **Prompt rules:** the LLM is told to answer only from the context and never make up policies.
- **Per-user memory:** every message is saved in MongoDB with a `user_id`. The last 6 messages are added to the prompt, so follow-up questions work ("what if I removed the bracelet links?").
- **Source citations:** each answer returns the knowledge base files it was based on.
- **Switchable LLM provider:** change `LLM_PROVIDER` in `.env` to use Gemini or OpenAI.
- **REST APIs** with FastAPI, auto-generated Swagger docs, and a simple chat UI for demos.

## Architecture

```
                         ┌──────────────────────────────┐
 Chat UI / API client ──►│  FastAPI   POST /chat        │
                         └──────────────┬───────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
 1. Load last N messages      2. Retrieve top-k chunks          3. Guardrail check
    (MongoDB memory)             (FAISS + embeddings)              score < threshold → refuse
                                        │
                                        ▼
              4. Prompt template = rules + context + history + question
                                        │
                                        ▼
                         5. LLM (Gemini / OpenAI) via LangChain
                                        │
                                        ▼
              6. Save question + answer to MongoDB → return {answer, sources}
```

**Ingestion (one time):** `data/knowledge_base/*.md` → split into chunks (600 chars, 100 overlap) → embeddings (`gemini-embedding-001`) → FAISS index saved in `faiss_index/`.

## Project structure

```
smartsupport/
├── app/
│   ├── main.py        # FastAPI app and endpoints
│   ├── chatbot.py     # Chat flow: memory → retrieval → guardrails → LLM → save
│   ├── rag.py         # Loads FAISS index and retrieves relevant chunks
│   ├── prompts.py     # Prompt template and guardrail replies
│   ├── memory.py      # MongoDB conversation history
│   ├── llm.py         # Gemini / OpenAI chat + embedding models
│   └── config.py      # Settings from .env
├── data/knowledge_base/   # Company documents (shipping, returns, warranty, ...)
├── static/index.html  # Chat UI
├── ingest.py          # Builds the FAISS index
├── requirements.txt
└── .env.example
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send `{ "user_id", "message" }` and get `{ "answer", "sources", "in_scope" }` |
| `GET` | `/history/{user_id}` | Get a user's full conversation |
| `DELETE` | `/history/{user_id}` | Clear a user's conversation memory |
| `GET` | `/health` | Server, MongoDB and provider status |
| `GET` | `/docs` | Interactive Swagger API docs |

Example:

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "utkarsha", "message": "Can I return a smartwatch?"}'
```

```json
{
  "answer": "Yes, smartwatches can be returned within 7 days of delivery...",
  "sources": ["returns_and_refunds.md"],
  "in_scope": true
}
```

## Setup and run

**Requirements:** Python 3.10+, MongoDB running locally (or a MongoDB Atlas URI), and a Gemini API key (free from [Google AI Studio](https://aistudio.google.com/apikey)).

```bash
# 1. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

# 2. Add your API key
copy .env.example .env          # macOS/Linux: cp .env.example .env
#    then edit .env and set GOOGLE_API_KEY

# 3. Build the FAISS index from the knowledge base
python ingest.py

# 4. Start the server
uvicorn app.main:app --port 8001
```

Open **http://localhost:8001** for the chat UI, or **http://localhost:8001/docs** for the API docs.

> If you change files in `data/knowledge_base/` or switch `LLM_PROVIDER`, run `python ingest.py` again, because the embeddings must come from the same model.

## Try these questions

| Question | What it shows |
|---|---|
| "Can I return a smartwatch?" | RAG answer with source |
| "What if I already removed links from the bracelet?" | Memory: understands the follow-up |
| "Do you offer no-cost EMI?" | Retrieval from payments doc |
| "Who won the cricket world cup?" | Guardrail: refused before the LLM is called |
| "Do you sell Rolex?" | Says it doesn't have that info instead of guessing |

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_CHAT_MODEL` | `gemini-3.5-flash` | Chat model |
| `TOP_K` | `4` | Chunks retrieved per question |
| `RELEVANCE_THRESHOLD` | `0.62` | Minimum cosine similarity to be in scope |
| `HISTORY_LIMIT` | `6` | Past messages sent as memory |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection |

## Possible improvements

- Streaming responses
- Rewriting follow-up questions into standalone questions before retrieval
- User authentication
- Evaluating answer quality with a test set
