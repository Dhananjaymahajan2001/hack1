from __future__ import annotations

import asyncio
import io
import inspect
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Ensure project root is importable on Streamlit Cloud and local runs.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.mcp_tools.retrieval_tools import RetrievalTools
from app.orchestration.graph import Orchestrator
from app.services.feedback_agent import FeedbackAgent
from app.services.sarvam_client import SarvamClient

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional fallback for quick setup
    PdfReader = None


st.set_page_config(page_title="Multilingual Policy Assistant", page_icon="📄", layout="wide")

st.title("Multilingual Scheme Document Intelligence")
st.caption("Chat + Session Memory + Feedback Agent + ChromaDB + Sarvam APIs")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
else:
    # Handle stale Streamlit session objects after hot-reload/code updates.
    try:
        run_params = inspect.signature(st.session_state.orchestrator.run).parameters
        if "feedback_context" not in run_params:
            st.session_state.orchestrator = Orchestrator()
    except Exception:
        st.session_state.orchestrator = Orchestrator()
if "retrieval" not in st.session_state:
    st.session_state.retrieval = RetrievalTools()
if "sarvam" not in st.session_state:
    st.session_state.sarvam = SarvamClient()
if "feedback_agent" not in st.session_state:
    st.session_state.feedback_agent = FeedbackAgent()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "feedback_events" not in st.session_state:
    st.session_state.feedback_events = []
if "message_feedback" not in st.session_state:
    st.session_state.message_feedback = {}
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "retry_requests" not in st.session_state:
    st.session_state.retry_requests = {}
if "voice_response_enabled" not in st.session_state:
    st.session_state.voice_response_enabled = False


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def detect_audio_mime(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return "audio/wav"
    if audio_bytes[:4] == b"RIFF":
        return "audio/wav"
    if audio_bytes[:3] == b"ID3" or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio/ogg"
    return "audio/wav"


def record_feedback(message: dict, rating: str, reason: str = "", source: str = "explicit") -> None:
    event = st.session_state.feedback_agent.make_feedback_event(
        message_id=message["id"],
        rating=rating,
        user_query=message.get("user_query", ""),
        assistant_answer=message.get("content", ""),
        reason=reason,
        source=source,
    )
    st.session_state.feedback_events.append(event)
    st.session_state.message_feedback[message["id"]] = rating


def get_conversation_context(max_turns: int = 6) -> list[dict]:
    context = []
    for msg in st.session_state.chat_messages[-max_turns:]:
        context.append({"role": msg["role"], "content": msg["content"]})
    return context


def generate_answer_audio(answer_text: str, selected_language_code: str) -> bytes:
    if not st.session_state.voice_response_enabled:
        return b""
    lang_for_tts = selected_language_code if selected_language_code != "auto" else "en-IN"
    audio_b64 = run_async(st.session_state.sarvam.text_to_speech(answer_text, lang_for_tts))
    return st.session_state.sarvam.decode_audio_base64(audio_b64)


def regenerate_with_feedback(message: dict, guidance: str, selected_language_code: str) -> None:
    retry_user_text = "Please try again."
    if guidance.strip():
        retry_user_text = f"Please try again with this preference: {guidance.strip()}"

    retry_user_message = {
        "id": str(uuid4()),
        "role": "user",
        "content": retry_user_text,
    }
    st.session_state.chat_messages.append(retry_user_message)

    feedback_context = st.session_state.feedback_agent.build_feedback_context(
        st.session_state.feedback_events,
        max_items=5,
    )
    conversation_context = get_conversation_context(max_turns=8)
    forced_lang = selected_language_code if selected_language_code != "auto" else None

    result = run_async(
        st.session_state.orchestrator.run(
            message.get("user_query", ""),
            forced_lang=forced_lang,
            feedback_context=feedback_context,
            conversation_context=conversation_context,
        )
    )
    answer = result.get("final_answer", "No answer generated.")
    citations = result.get("citations", [])
    answer_audio = generate_answer_audio(answer, selected_language_code)

    retry_assistant_message = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "user_query": message.get("user_query", ""),
        "audio_bytes": answer_audio,
    }
    st.session_state.chat_messages.append(retry_assistant_message)
    st.session_state.retry_requests[message["id"]] = False


with st.sidebar:
    st.subheader("Language")
    language_options = {
        "Auto Detect": "auto",
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Dogri": "doi-IN",
    }
    selected_language_label = st.selectbox("Input language", options=list(language_options.keys()), index=0)
    selected_language_code = language_options[selected_language_label]

    st.divider()
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index Uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        elif PdfReader is None:
            st.error("Install pypdf to enable PDF parsing.")
        else:
            indexed = 0
            for f in uploaded_files:
                data = f.read()
                reader = PdfReader(io.BytesIO(data))
                full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                chunks = chunk_text(full_text)
                if not chunks:
                    continue
                ids = [str(uuid4()) for _ in chunks]
                metas = [
                    {"doc_name": f.name, "section": f"chunk_{idx+1}", "source": "upload"}
                    for idx, _ in enumerate(chunks)
                ]
                st.session_state.retrieval.upsert_chunks(ids, chunks, metas)
                indexed += len(chunks)
            st.success(f"Indexed {indexed} chunks into ChromaDB.")

    st.divider()
    st.subheader("Voice Input")
    st.session_state.voice_response_enabled = st.toggle(
        "Voice response mode",
        value=st.session_state.voice_response_enabled,
        help="When enabled, assistant answers are also returned as audio.",
    )
    stt_mode_options = ["transcribe", "translate", "verbatim", "translit", "codemix"]
    default_mode = st.session_state.sarvam.settings.sarvam_stt_mode
    default_mode_idx = stt_mode_options.index(default_mode) if default_mode in stt_mode_options else 0
    selected_stt_mode = st.selectbox(
        "Speech mode",
        options=stt_mode_options,
        index=default_mode_idx,
        help="Use codemix for Hindi-English mixed speech.",
    )
    audio = st.audio_input("Speak your query")
    if st.button("Transcribe Voice", use_container_width=True):
        if not audio:
            st.warning("Record audio first.")
        else:
            try:
                audio_bytes = audio.read()
                stt_lang = selected_language_code if selected_language_code != "auto" else "auto"
                transcript = run_async(
                    st.session_state.sarvam.speech_to_text(
                        audio_bytes,
                        stt_lang,
                        filename=getattr(audio, "name", "audio.wav"),
                        mode=selected_stt_mode,
                    )
                )
                if transcript.strip():
                    st.session_state.voice_transcript = transcript
                    st.success("Transcription complete.")
                else:
                    st.warning("No transcript returned. Try again, or switch speech mode to codemix/transcribe.")
            except Exception as exc:
                st.error(f"Voice transcription failed: {exc}")

    st.divider()
    sat = len([e for e in st.session_state.feedback_events if e.get("rating") == "satisfied"])
    unsat = len([e for e in st.session_state.feedback_events if e.get("rating") == "unsatisfied"])
    st.caption(f"Feedback summary -> Satisfied: {sat} | Unsatisfied: {unsat}")
    if st.button("Reset Chat Session", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.feedback_events = []
        st.session_state.message_feedback = {}
        st.session_state.voice_transcript = ""
        st.rerun()


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        citations = message.get("citations", [])
        audio_bytes = message.get("audio_bytes", b"")
        if message["role"] == "assistant" and citations:
            with st.expander("Citations"):
                for c in citations:
                    st.write(f"- {c}")
        if message["role"] == "assistant" and audio_bytes:
            st.audio(audio_bytes, format=detect_audio_mime(audio_bytes))

        if message["role"] == "assistant":
            feedback_key = f"feedback_choice_{message['id']}"
            saved_rating = st.session_state.message_feedback.get(message["id"], "")
            options = ["", "satisfied", "unsatisfied"]
            default_idx = options.index(saved_rating) if saved_rating in options else 0
            chosen = st.radio(
                "Was this response helpful?",
                options=options,
                index=default_idx,
                key=feedback_key,
                horizontal=True,
                format_func=lambda x: {
                    "": "Not rated",
                    "satisfied": "👍 Satisfied",
                    "unsatisfied": "👎 Not satisfied",
                }[x],
            )
            if chosen and chosen != saved_rating:
                reason = "Manual rating from user"
                record_feedback(message, chosen, reason=reason, source="explicit")
                if chosen == "unsatisfied":
                    st.session_state.retry_requests[message["id"]] = True
                st.success("Feedback captured for this response.")

            if st.session_state.retry_requests.get(message["id"], False):
                st.info("Would you like to try again? Tell me how to improve (for example: shorter, bullet points, simpler Hindi).")
                guidance_key = f"retry_guidance_{message['id']}"
                guidance = st.text_input("Retry instruction", key=guidance_key)
                retry_clicked = st.button("Try again with this feedback", key=f"retry_btn_{message['id']}")
                if retry_clicked:
                    if guidance.strip():
                        record_feedback(
                            message,
                            "unsatisfied",
                            reason=f"Retry guidance: {guidance.strip()}",
                            source="explicit_retry",
                        )
                    with st.spinner("Regenerating improved response..."):
                        regenerate_with_feedback(message, guidance, selected_language_code)
                    st.rerun()


chat_placeholder = "Ask about schemes/policies/circulars..."
if st.session_state.voice_transcript:
    chat_placeholder = f"Voice transcript ready: {st.session_state.voice_transcript[:80]}"

user_query = st.chat_input(chat_placeholder)
if not user_query and st.session_state.voice_transcript:
    user_query = st.session_state.voice_transcript
    st.session_state.voice_transcript = ""

if user_query:
    user_message = {
        "id": str(uuid4()),
        "role": "user",
        "content": user_query.strip(),
    }
    st.session_state.chat_messages.append(user_message)

    if st.session_state.feedback_agent.detect_frustration(user_query):
        pseudo_assistant = {
            "id": str(uuid4()),
            "content": "Implicit frustration detected from user text.",
            "user_query": user_query,
        }
        record_feedback(
            pseudo_assistant,
            rating="unsatisfied",
            reason="Frustration keywords detected in user message",
            source="implicit",
        )

    feedback_context = st.session_state.feedback_agent.build_feedback_context(
        st.session_state.feedback_events,
        max_items=5,
    )
    conversation_context = get_conversation_context(max_turns=8)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with multilingual RAG + feedback memory..."):
            forced_lang = selected_language_code if selected_language_code != "auto" else None
            result = run_async(
                st.session_state.orchestrator.run(
                    user_query.strip(),
                    forced_lang=forced_lang,
                    feedback_context=feedback_context,
                    conversation_context=conversation_context,
                )
            )
        answer = result.get("final_answer", "No answer generated.")
        citations = result.get("citations", [])
        answer_audio = generate_answer_audio(answer, selected_language_code)
        st.markdown(answer)
        if citations:
            with st.expander("Citations"):
                for c in citations:
                    st.write(f"- {c}")
        if answer_audio:
            st.audio(answer_audio, format=detect_audio_mime(answer_audio))
        elif st.session_state.voice_response_enabled:
            st.caption("Voice response requested, but TTS audio was not returned for this message.")

    assistant_message = {
        "id": str(uuid4()),
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "user_query": user_query.strip(),
        "audio_bytes": answer_audio,
    }
    st.session_state.chat_messages.append(assistant_message)
    st.rerun()
