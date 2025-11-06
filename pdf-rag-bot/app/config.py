# app/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # Provider selection
    PROVIDER: str = "gemini"  # openai | azure | gemini

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-large"

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_CHAT_DEPLOYMENT: str | None = None
    AZURE_OPENAI_EMBED_DEPLOYMENT: str | None = None

    # Gemini
    GOOGLE_API_KEY: str | None = None
    GEMINI_CHAT_MODEL: str = "gemini-2.5-pro"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    # RAG params
    PERSIST_DIR: str = str(Path("./data/store").resolve())      # generic fallback
    UPLOAD_DIR: str = str(Path("./data/uploads").resolve())
    TOP_K: int = 6
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # LanceDB params
    LANCE_DIR: str = str(Path("./.data/lancedb").resolve())     # good for Streamlit Cloud
    LANCE_TABLE: str = "pdf_rag"

    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",           # also let pydantic read .env
        case_sensitive=False,
    )

def _maybe_override_from_streamlit_secrets(cfg: "Settings") -> None:
    """
    If running on Streamlit Cloud, read keys from st.secrets and
    override env/.env values. Safe no-op locally.
    """
    try:
        import streamlit as st
        # Only override if present in st.secrets
        if "OPENAI_API_KEY" in st.secrets:
            cfg.OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        if "GOOGLE_API_KEY" in st.secrets:
            cfg.GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

        # Azure (optional)
        if "AZURE_OPENAI_API_KEY" in st.secrets:
            cfg.AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
        if "AZURE_OPENAI_ENDPOINT" in st.secrets:
            cfg.AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
        if "AZURE_OPENAI_CHAT_DEPLOYMENT" in st.secrets:
            cfg.AZURE_OPENAI_CHAT_DEPLOYMENT = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        if "AZURE_OPENAI_EMBED_DEPLOYMENT" in st.secrets:
            cfg.AZURE_OPENAI_EMBED_DEPLOYMENT = st.secrets["AZURE_OPENAI_EMBED_DEPLOYMENT"]

        # Optional: allow provider override via secrets
        if "PROVIDER" in st.secrets:
            cfg.PROVIDER = st.secrets["PROVIDER"]
    except Exception:
        pass

# Instantiate and finalize settings
settings = Settings()
_maybe_override_from_streamlit_secrets(settings)

# Ensure dirs exist (works both locally & on Streamlit Cloud)
Path(settings.PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.LANCE_DIR).mkdir(parents=True, exist_ok=True)
