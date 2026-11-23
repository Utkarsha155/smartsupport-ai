"""The main chat flow: memory -> retrieval -> guardrails -> prompt -> LLM -> save."""
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app import config, memory, rag
from app.llm import get_chat_model
from app.prompts import CHAT_PROMPT, GREETING_REPLY, OUT_OF_SCOPE_REPLY, THANKS_REPLY

# LangChain "chain": prompt template -> chat model -> plain string
_chain = CHAT_PROMPT | get_chat_model() | StrOutputParser()

GREETING_PATTERN = re.compile(r"^(hi+|hello+|hey+|hii+|good (morning|afternoon|evening)|namaste)\W*$", re.I)
THANKS_PATTERN = re.compile(r"^(thanks?|thank you|thx|ok(ay)?|great|cool)\W*( so much| a lot)?\W*$", re.I)


def _to_langchain_history(messages: list[dict]) -> list:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in messages
    ]


def _find_context(message: str, history: list[dict]):
    """
    Returns (relevant chunks, guardrail score).

    - The guardrail score is based ONLY on the user's current message, so an
      off-topic question can never pass just because the previous one was on-topic.
    - For the actual context, a follow-up like "what if I removed the links?" is
      searched together with the previous question so the right chunks are found.
    """
    results = rag.retrieve(message)
    best_score = results[0][1] if results else 0.0
    if best_score < config.RELEVANCE_THRESHOLD:
        return [], best_score

    previous_questions = [m["content"] for m in history if m["role"] == "user"]
    if previous_questions:
        results = rag.retrieve(f"{previous_questions[-1]} {message}")

    docs = [doc for doc, score in results if score >= config.RELEVANCE_THRESHOLD]
    return docs or [results[0][0]], best_score


def get_reply(user_id: str, message: str) -> dict:
    message = message.strip()
    history = memory.get_recent_messages(user_id)

    sources: list[str] = []
    in_scope = True

    # Guardrail 1: small talk gets a fixed reply (no LLM call needed)
    if GREETING_PATTERN.match(message):
        answer = GREETING_REPLY
    elif THANKS_PATTERN.match(message):
        answer = THANKS_REPLY
    else:
        docs, best_score = _find_context(message, history)

        # Guardrail 2: nothing in the knowledge base is close enough -> out of scope
        if best_score < config.RELEVANCE_THRESHOLD:
            answer = OUT_OF_SCOPE_REPLY
            in_scope = False
        else:
            sources = sorted({doc.metadata["source"] for doc in docs})

            # Guardrail 3: the prompt itself tells the LLM to answer only from context
            answer = _chain.invoke(
                {
                    "context": rag.format_context(docs),
                    "history": _to_langchain_history(history),
                    "question": message,
                }
            ).strip()

    memory.save_message(user_id, "user", message)
    memory.save_message(user_id, "assistant", answer)

    return {"answer": answer, "sources": sources, "in_scope": in_scope}
