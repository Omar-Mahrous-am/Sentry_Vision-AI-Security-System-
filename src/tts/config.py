"""
Configuration for the Text-to-Speech (TTS) module.

Centralizes all TTS parameters: language, repetitions,
output directory, and audio format settings.
"""

import os


class TTSConfig:
    """Configuration class for the TTS engine."""

    # --- Audio Output ---
    OUTPUT_DIR = os.path.join("data", "tts_output")
    DEFAULT_FILENAME = "emergency_alert.mp3"

    # --- gTTS Settings ---
    LANGUAGE = "en"
    SLOW = False  # Normal speed playback

    # --- Alert Behaviour ---
    REPEAT_COUNT = 4  # Number of times the alert message is repeated


# Singleton instance for easy import
tts_config = TTSConfig()
