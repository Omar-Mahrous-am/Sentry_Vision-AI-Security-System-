#!/usr/bin/env python
"""
Standalone CLI script for the Sentry Vision TTS module.

Usage examples
--------------
  # Basic — synthesize a fire-alert message
  python scripts/run_tts.py --text "There is fire in Area 3 near Gate 14, please call Emergency!"

  # Custom output file, repeat 6 times, and play after saving
  python scripts/run_tts.py --text "Stolen car detected near Gate 4" --output alert.mp3 --repeat 6 --play
"""

import argparse
import sys
import os

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tts import synthesize, play_alert


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentry Vision — Text-to-Speech alert generator"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The alert message to convert to speech.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output MP3 file (default: data/tts_output/emergency_alert.mp3).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Number of times to repeat the message (default: 4).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help='Language code, e.g. "en", "ar" (default: en).',
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        default=False,
        help="Generate speech at a slower tempo.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        default=False,
        help="Play the audio after saving.",
    )

    args = parser.parse_args()

    saved_path = synthesize(
        text=args.text,
        output_path=args.output,
        repeat=args.repeat,
        lang=args.lang,
        slow=args.slow,
    )

    print(f"\n✅ Audio saved → {saved_path}")
    print("-" * 40)

    if args.play:
        print("▶  Playing audio …")
        play_alert(saved_path)


if __name__ == "__main__":
    main()
