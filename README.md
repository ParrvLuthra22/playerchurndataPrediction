# 🎮 Player Churn Prediction & Retention Advisor

A production-ready Streamlit application that combines:

- **ML inference** for churn prediction (using a pre-trained `churn_model.pkl`)
- **RAG retrieval** from retention strategy knowledge (`rag/knowledge.txt`)
- **LangGraph workflow** to generate concise, actionable retention recommendations

This repository is structured so prediction works independently, while GenAI/RAG features degrade gracefully if API configuration is unavailable.

---

## Key capabilities

1. Upload a CSV and run churn prediction across all rows.
2. Validate uploaded schema against model-required features.
3. Display prediction output table + predicted churn rate.
4. Run AI advisor for a sample row (prioritizes high-risk player when available).
5. Retrieve supporting retention snippets from knowledge base for transparent recommendations.

---

## Project structure

```text
playerchurndataPrediction/
├── app.py                     # Streamlit UI + prediction + advisor orchestration
├── agent/
│   ├── __init__.py
│   └── graph.py               # LangGraph nodes (analyze, retrieve, generate)
├── rag/
│   ├── __init__.py
│   ├── knowledge.txt          # Domain retention strategy corpus
│   ├── vector_store.py        # Chroma index build + retrieval + fallback retrieval
│   └── chroma_db/             # Persisted Chroma database
├── utils/
│   ├── __init__.py
│   └── llm.py                 # Groq/OpenAI-compatible client configuration
├── model/
│   └── churn_model.pkl        # Primary model path
├── assets/models/
│   └── churn_model.pkl        # Backward-compatible fallback model path
├── requirements.txt
└── README.md
```

---

## Model input contract

The current trained model expects these feature columns:

- `Age`
- `Gender`
- `Location`
- `GameGenre`
- `PlayTimeHours`
- `InGamePurchases`
- `GameDifficulty`
- `SessionsPerWeek`
- `AvgSessionDurationMinutes`
- `PlayerLevel`
- `AchievementsUnlocked`

Columns like `PlayerID`, `EngagementLevel`, and `Churn` are treated as non-feature metadata and removed before prediction.

---

## Setup

### 1) Create and activate a Python environment

Use your preferred environment manager (`venv`, `conda`, etc.).

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables (for GenAI/RAG features)

Set in shell or `.env`:

- `GROQ_API_KEY` (**required** for AI advisor)
- `GROQ_BASE_URL` (optional, default: `https://api.groq.com/openai/v1`)
- `GROQ_MODEL` (optional, default: `llama-3.1-8b-instant`)
- `GROQ_EMBEDDING_MODEL` (optional, default: `text-embedding-3-small`)
- `GROQ_TEMPERATURE` (optional, default: `0.25`)

### 4) Ensure model file exists

Supported locations:

- `model/churn_model.pkl` (preferred)
- `assets/models/churn_model.pkl` (fallback)

### 5) Run the app

```bash
streamlit run app.py
```

---

## Architecture overview

### `app.py` (application layer)

- Loads model from supported paths.
- Preprocesses uploaded CSV.
- Validates schema and aligns columns to model feature order.
- Runs prediction and displays metrics.
- Invokes AI advisor workflow for a sample player.

### `agent/graph.py` (reasoning/orchestration layer)

- Implements a 3-step LangGraph pipeline:
  - `analyze`: builds player risk context
  - `retrieve`: fetches relevant strategy snippets
  - `generate`: returns explanation + recommendations via LLM
- Uses cached compiled graph for efficient repeated execution.

### `rag/vector_store.py` (knowledge retrieval layer)

- Builds/loads Chroma vector store from `knowledge.txt`.
- Retrieves top-k relevant chunks.
- Provides keyword-overlap fallback when embeddings are unavailable.

### `utils/llm.py` (model client configuration)

- Centralizes LLM settings and client creation.
- Ensures consistent chat/embedding config across the app.

---

## Troubleshooting

### Prediction fails after upload

- Check the **Model Schema Check** panel in the UI.
- Ensure all required model feature columns are present.
- Remove or rename non-matching columns to match expected training schema.

### App runs but advisor fails

- Verify `GROQ_API_KEY` is set.
- Confirm internet access and model/embedding settings.
- Prediction remains available even if advisor features are unavailable.

### Model not found

- Confirm `churn_model.pkl` exists in either `model/` or `assets/models/`.

---

## Engineering notes

- Inference-only application (no retraining inside app runtime)
- Clear module boundaries (UI, orchestration, retrieval, LLM config)
- Path-safe file handling with `pathlib`
- Defensive error handling and user-facing diagnostics
