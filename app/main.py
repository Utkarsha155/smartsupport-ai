"""FastAPI app: exposes the chat APIs and serves the demo chat page."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import chatbot, config, memory, rag

logger = logging.getLogger("smartsupport")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: connect to MongoDB and load the FAISS index into memory
    memory.init_db()
    rag.load_vector_store()
    yield


app = FastAPI(
    title="SmartSupport API",
    description=f"AI customer support chatbot for {config.COMPANY_NAME} (RAG + MongoDB memory)",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Request / response models ----------
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, examples=["utkarsha"])
    message: str = Field(..., min_length=1, max_length=1000, examples=["What is your return policy?"])


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    in_scope: bool


class Message(BaseModel):
    role: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    user_id: str
    messages: list[Message]


# ---------- Endpoints ----------
@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(config.BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mongodb": memory.is_connected(), "llm_provider": config.LLM_PROVIDER}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    try:
        return chatbot.get_reply(request.user_id.strip(), request.message)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again in a moment.",
        )


@app.get("/history/{user_id}", response_model=HistoryResponse)
def get_history(user_id: str):
    messages = [
        {**m, "created_at": m["created_at"].isoformat()} for m in memory.get_all_messages(user_id)
    ]
    return {"user_id": user_id, "messages": messages}


@app.delete("/history/{user_id}")
def delete_history(user_id: str):
    deleted = memory.clear_messages(user_id)
    return {"user_id": user_id, "deleted_messages": deleted}
