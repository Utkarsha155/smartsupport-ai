"""Creates the chat model and embedding model for the selected provider (Gemini or OpenAI)."""
from app import config


def get_chat_model():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2,
            max_retries=2,
            timeout=60,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.2,  # low temperature = factual, consistent answers
        thinking_budget=0,  # turn off "thinking" mode: support answers don't need it, and it is much faster
        max_retries=2,      # fail fast instead of hanging if the API is down or rate-limited
        timeout=90,         # seconds to wait for a reply
    )


def get_embeddings():
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
