import os
import streamlit as st


def get_secret(name: str, default=None):
    try:
        value = st.secrets(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)


APP_NAME = "AI Education / AI Tutor"
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL")
