from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from sarvamai import SarvamAI
except Exception:  # pragma: no cover
    SarvamAI = None


class SarvamClient:
    """
    Thin Sarvam API wrapper.
    Uses graceful fallbacks so the app remains runnable during local development.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.sarvam_api_subscription_key or self.settings.sarvam_api_key
        self.client = None
        if self.api_key and SarvamAI is not None:
            try:
                self.client = SarvamAI(api_subscription_key=self.api_key)
            except Exception as exc:
                logger.warning("Sarvam SDK init failed, running in fallback mode: %s", exc)

    async def detect_language(self, text: str) -> str:
        if not self.client:
            return self._heuristic_lang(text)

        try:
            response = await asyncio.to_thread(self.client.text.identify_language, input=text)
            data = self._to_dict(response)
            return data.get("language_code", self._heuristic_lang(text))
        except Exception as exc:
            logger.warning("Language detect failed, using heuristic: %s", exc)
            return self._heuristic_lang(text)

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text

        if not self.client:
            return text

        try:
            response = await asyncio.to_thread(
                self.client.text.translate,
                input=text,
                source_language_code=source_lang or "auto",
                target_language_code=target_lang,
                speaker_gender=self.settings.sarvam_speaker_gender,
            )
            data = self._to_dict(response)
            return data.get("translated_text", text)
        except Exception as exc:
            logger.warning("Translation failed, returning original text: %s", exc)
            return text

    async def chat_completion(
        self,
        prompt: str,
        context_chunks: list[str],
        feedback_context: str = "",
        conversation_context: list[dict] | None = None,
    ) -> str:
        if not self.client:
            return self._mock_answer(prompt, context_chunks)

        convo_lines = []
        if conversation_context:
            for item in conversation_context[-6:]:
                role = item.get("role", "user")
                content = item.get("content", "")
                if content:
                    convo_lines.append(f"{role}: {content}")
        convo_block = "\n".join(convo_lines)

        user_content = (
            f"Question: {prompt}\n\n"
            + ("Recent Conversation:\n" + convo_block + "\n\n" if convo_block else "")
            + ("Feedback Guidance:\n" + feedback_context + "\n\n" if feedback_context else "")
            + "Context:\n"
            + "\n\n".join(context_chunks)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a policy assistant. Use provided context only. "
                    "Return concise, accurate responses with citations."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        try:
            kwargs: dict[str, Any] = {"messages": messages}
            if self.settings.sarvam_chat_model:
                kwargs["model"] = self.settings.sarvam_chat_model
            response = await asyncio.to_thread(self.client.chat.completions, **kwargs)
        except Exception as exc:
            logger.error("Sarvam chat failed, using local fallback: %s", exc)
            return self._mock_answer(prompt, context_chunks)

        data = self._to_dict(response)
        choices = data.get("choices", [])
        if not choices:
            return "I could not find an answer in the current document context."
        first = choices[0]
        if isinstance(first, dict):
            return first.get("message", {}).get("content", "").strip()
        return str(first)

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        language_code: str = "auto",
        filename: str = "audio.wav",
        mode: str | None = None,
    ) -> str:
        if not self.client:
            return ""

        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename
            stt_mode = mode or self.settings.sarvam_stt_mode
            kwargs: dict[str, Any] = {
                "file": audio_file,
                "model": self.settings.sarvam_stt_model,
                "mode": stt_mode,
            }
            if language_code != "auto":
                kwargs["language_code"] = language_code
            response = await asyncio.to_thread(self.client.speech_to_text.transcribe, **kwargs)
            data = self._to_dict(response)
            return data.get("transcript", "")
        except Exception as exc:
            logger.warning("STT failed: %s", exc)
            return ""

    async def text_to_speech(self, text: str, language_code: str) -> str:
        if not self.client:
            return ""

        try:
            response = await asyncio.to_thread(
                self.client.text_to_speech.convert,
                text=text,
                target_language_code=language_code,
            )
            data = self._to_dict(response)
            return self._extract_audio_base64(data)
        except Exception as exc:
            logger.warning("TTS failed: %s", exc)
            return ""

    def decode_audio_base64(self, audio_b64: str) -> bytes:
        if not audio_b64:
            return b""
        try:
            return base64.b64decode(audio_b64)
        except Exception:
            return b""

    def _to_dict(self, response: Any) -> dict[str, Any]:
        if response is None:
            return {}
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass
        if hasattr(response, "dict"):
            try:
                return response.dict()
            except Exception:
                pass
        if hasattr(response, "__dict__"):
            return dict(response.__dict__)
        return {}

    def _extract_audio_base64(self, data: dict[str, Any]) -> str:
        direct = data.get("audio_base64")
        if isinstance(direct, str) and direct:
            return direct

        audio_data = data.get("audio")
        if isinstance(audio_data, str) and audio_data:
            return audio_data

        outputs = data.get("outputs")
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                val = first.get("audio_base64") or first.get("audio")
                if isinstance(val, str):
                    return val
        return ""

    def _heuristic_lang(self, text: str) -> str:
        devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
        if devanagari:
            return "hi-IN"
        return "en-IN"

    def _mock_answer(self, prompt: str, chunks: list[str]) -> str:
        context = chunks[0][:600] if chunks else "No context found in documents."
        return (
            f"Draft answer for: {prompt}\n\n"
            f"Top matched context:\n{context}\n\n"
            "Note: This is a local mock response. Add SARVAM_API_KEY or SARVAM_API_SUBSCRIPTION_KEY in .env."
        )
