from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_api_subscription_key: str = Field(default="", alias="SARVAM_API_SUBSCRIPTION_KEY")
    # Backward-compatible legacy field; SDK client does not require base URL override.
    sarvam_base_url: str = Field(default="https://api.sarvam.ai", alias="SARVAM_BASE_URL")
    sarvam_chat_model: str = Field(default="", alias="SARVAM_CHAT_MODEL")
    sarvam_stt_model: str = Field(default="saaras:v3", alias="SARVAM_STT_MODEL")
    sarvam_stt_mode: str = Field(default="transcribe", alias="SARVAM_STT_MODE")
    sarvam_speaker_gender: str = Field(default="Male", alias="SARVAM_SPEAKER_GENDER")

    chroma_persist_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="gov_docs", alias="CHROMA_COLLECTION")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    metadata_sample_size: int = Field(default=5000, alias="METADATA_SAMPLE_SIZE")

    default_language: str = Field(default="en", alias="APP_DEFAULT_LANGUAGE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
