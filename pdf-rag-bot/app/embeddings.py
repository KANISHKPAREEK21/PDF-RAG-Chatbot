from __future__ import annotations
import os
from app.config import settings

from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _normalize_gemini_model(name: str | None) -> str:
    # Gemini expects resource form: "models/text-embedding-004"
    name = (name or "models/text-embedding-004").strip()
    if not name.startswith("models/"):
        name = f"models/{name}"
    return name


def get_embeddings():
    provider = (settings.PROVIDER or "openai").lower()

    if provider == "gemini":
        # Requires GOOGLE_API_KEY in env
        os.environ.setdefault("GOOGLE_API_KEY", settings.GOOGLE_API_KEY or "")
        model = _normalize_gemini_model(settings.GEMINI_EMBED_MODEL)
        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "azure":
        # For Azure you must have a deployment name, not a model name
        # settings.AZURE_OPENAI_EMBED_DEPLOYMENT should be set in .env
        return AzureOpenAIEmbeddings(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_EMBED_DEPLOYMENT,
            # api_version is optional with the latest SDK, add if your resource needs it:
            # openai_api_version="2024-05-01-preview",
        )

    # default: OpenAI
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBED_MODEL or "text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY,
    )
