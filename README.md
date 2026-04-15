# Multilingual Scheme Document Intelligence - Starter

This starter implements the orchestration layer between:
- Streamlit chat UI (session state + feedback capture)
- ChromaDB vector retrieval (MCP-style tools)
- Sarvam APIs for language + generation + voice
- LangGraph stateful workflow

## Current Flow
1. User asks via text (or voice transcription)
2. Detect language (`hi-IN`, `en-IN`, etc.) or use manual override (`doi-IN` supported in UI)
3. Translate to pivot language (`en-IN`)
4. Retrieve chunks from ChromaDB
   - Agentic retriever does query expansion + reciprocal rank fusion (RRF)
   - Optional LLM-based metadata filter extraction (if Azure env vars are provided)
5. Generate grounded answer using retrieved context
6. Translate answer back to user language
7. Persist feedback (satisfied/unsatisfied + implicit frustration signals)
8. Feed feedback context into next turn for response improvement
9. Return answer + citations

## Project Structure
```text
app/
  config/settings.py
  mcp_tools/retrieval_tools.py
  orchestration/graph.py
  services/sarvam_client.py
  streamlit_app.py
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Sarvam key in `.env`:
```env
SARVAM_API_SUBSCRIPTION_KEY=your_key_here
# fallback supported:
SARVAM_API_KEY=
SARVAM_CHAT_MODEL=
SARVAM_STT_MODEL=saaras:v3
SARVAM_STT_MODE=transcribe
SARVAM_SPEAKER_GENDER=Male
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=gov_docs
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
METADATA_SAMPLE_SIZE=5000
GPT4o_API_KEY=optional_for_agentic_retrieval
GPT4o_DEPLOYMENT_ENDPOINT=optional_for_agentic_retrieval
```

## Run
```bash
streamlit run app/streamlit_app.py
```

## Notes
- Sarvam integration now uses the official `sarvamai` SDK client pattern.
- Without `SARVAM_API_SUBSCRIPTION_KEY` (or `SARVAM_API_KEY`), the app runs in mock mode for fast local development.
- PDF ingestion is included for quick indexing into ChromaDB.
- Streamlit language selector includes Dogri (`doi-IN`) for explicit routing.
- Retriever auto-falls back to Chroma text-query mode if the sentence-transformer model cannot be downloaded.
