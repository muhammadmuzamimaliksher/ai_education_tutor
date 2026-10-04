import os
import streamlit as st


def get_secret(name: str, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name, default)

def get_api_key(GROQ_API_KEY):
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


APP_NAME = "AI Education / AI Tutor"
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
