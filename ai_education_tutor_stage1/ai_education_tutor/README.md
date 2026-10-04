# 🎓 AI Education / AI Tutor — Stage 1

A beginner-friendly foundation for an AI education platform using:

- Python
- Streamlit
- Groq API
- `openai/gpt-oss-120b`

## Stage 1 architecture

Streamlit UI → Groq API → AI Tutor → Answer

Later stages will add:

1. CrewAI agents
2. RAG and vector database
3. Source-grounded answers
4. Verification agent
5. Adaptive Tutor
6. CrewAI Memory
7. Calculator and web tools
8. Quiz/practice system
9. Testing and error handling
10. GitHub
11. Streamlit Cloud deployment

## Python version

Use Python 3.12 for this project.

## Local installation

### 1. Create a virtual environment

Windows CMD:

```text
python -m venv .venv
.venv\Scripts\activate
```

PowerShell:

```text
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure your Groq API key

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Never upload `secrets.toml` to GitHub.

### 4. Run the app

```text
streamlit run app.py
```

## Test questions

### Test 1 — Basic

Settings:
- Academic Level: Grade 1–5
- Subject: Science
- Language: English
- Explanation Style: Simple

Question:

```text
Explain photosynthesis.
```

### Test 2 — Mathematics

Settings:
- Academic Level: Grade 9–10
- Subject: Mathematics
- Language: English
- Explanation Style: Step-by-step

Question:

```text
Solve 2x² + 5x - 3 = 0 step by step.
```

### Test 3 — Urdu

Settings:
- Academic Level: Grade 6–8
- Subject: Science
- Language: Urdu
- Explanation Style: Simple

Question:

```text
Explain photosynthesis.
```

## Common errors

### Invalid API key

Check:

```toml
GROQ_API_KEY = "your-real-key"
```

Make sure there are no extra spaces or accidental quotes inside the key.

### Model error / 404

Check that the model configured in `secrets.toml` is available to your Groq account.

### ModuleNotFoundError: groq

Run:

```text
pip install groq
```

or reinstall:

```text
pip install -r requirements.txt
```

### Streamlit command not found

Run:

```text
python -m streamlit run app.py
```

## Project structure

```text
ai_education_tutor/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── agents/
│   └── __init__.py
├── rag/
│   └── __init__.py
├── tools/
│   └── __init__.py
├── memory/
│   └── __init__.py
├── services/
│   └── __init__.py
├── data/
│   └── knowledge/
│
└── .streamlit/
    ├── config.toml
    └── secrets.toml
```
