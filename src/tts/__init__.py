"""Sentry Vision — Text-to-Speech (TTS) sub-package."""

from .config import tts_config
from .tts_engine import generate_alert_audio

__all__ = ["tts_config", "generate_alert_audio"]
