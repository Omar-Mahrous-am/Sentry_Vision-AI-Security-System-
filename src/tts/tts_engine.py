import pyttsx3
import threading

tts_lock = threading.Lock()

def generate_alert_audio(text: str) -> None:
    """
    Speaks the alert message out loud on the server machine.
    No file saving, no URL returned.
    """
    def speak():
        with tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)   # speaking speed
                engine.setProperty('volume', 1.0) # max volume
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception as e:
                print(f"TTS error: {e}")

    # Run in a background thread so it doesn't block the API response
    threading.Thread(target=speak, daemon=True).start()