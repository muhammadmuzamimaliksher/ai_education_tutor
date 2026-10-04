import os
import streamlit as st


# =========================================================
# Application
# =========================================================

APP_TITLE = "AI Education / AI Tutor"


# =========================================================
# Groq API Key
# =========================================================

def get_groq_api_key():
    """
    Get Groq API key.

    Priority:
    1. Streamlit Secrets
    2. Environment variable
    """

    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")

        if secret_key:
            return str(secret_key).strip()

    except Exception:
        pass

    environment_key = os.getenv("GROQ_API_KEY", "")

    return environment_key.strip()


GROQ_API_KEY = get_groq_api_key()


# =========================================================
# Models
# =========================================================

DEFAULT_MODEL = "llama-3.3-70b-versatile"

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


# =========================================================
# Optional Model Override
# =========================================================

try:
    secret_model = st.secrets.get("GROQ_MODEL", "")

    if secret_model:
        DEFAULT_MODEL = str(secret_model).strip()

        if DEFAULT_MODEL not in AVAILABLE_MODELS:
            AVAILABLE_MODELS.insert(0, DEFAULT_MODEL)

except Exception:
    pass
