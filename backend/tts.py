import pyttsx3
import os
import uuid
import threading

# pyttsx3 can be tricky in multi-threaded environments like FastAPI
# Creating a lock to avoid concurrent engine runs causing errors
tts_lock = threading.Lock()

def generate_alert_audio(text: str, output_directory: str = "audio_outputs") -> str:
    """
    Generates text-to-speech audio and saves it to a file.
    Returns the file path.
    """
    # Create the output directory relative to the current working directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate a unique file name to avoid collisions
    filename = f"alert_{uuid.uuid4().hex}.wav"
    filepath = os.path.join(output_directory, filename)
    
    with tts_lock:
        try:
            # Initialize engine locally to prevent loop issues
            engine = pyttsx3.init()
            
            # Configure properties (optional)
            # engine.setProperty('rate', 150)
            
            engine.save_to_file(text, filepath)
            engine.runAndWait()
            
            # Clean up engine in multithreaded context to avoid Com errors
            del engine
            
            import wave
            import struct
            import math
            
            # --- Add Alarm Sound Natively ---
            # Read the generated speech params
            with wave.open(filepath, 'rb') as w:
                params = w.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                speech_frames = w.readframes(nframes)

            # Generate alarm matching those params
            alarm_frames = bytearray()
            duration = 1.5 # 1.5 second alarm
            freq = 800.0
            for i in range(int(framerate * duration)):
                # sine wave
                value = int(16000 * math.sin(2.0 * math.pi * freq * i / framerate))
                # pack based on sampwidth
                if sampwidth == 2:
                    alarm_frames.extend(struct.pack('<h', value))
                elif sampwidth == 1:
                    alarm_frames.extend(struct.pack('<B', int((value + 32768) / 256)))
                    
            # Duplicate for multiple channels if stereo
            if nchannels == 2:
                stereo_alarm = bytearray()
                for i in range(0, len(alarm_frames), sampwidth):
                    sample = alarm_frames[i:i+sampwidth]
                    stereo_alarm.extend(sample)
                    stereo_alarm.extend(sample)
                alarm_frames = stereo_alarm

            # Export the combined file back to the same path
            # Combine them: message -> alarm depending on preference, we'll do message then alarm
            with wave.open(filepath, 'wb') as w:
                w.setparams(params)
                w.writeframes(speech_frames + alarm_frames)
            
        except Exception as e:
            print(f"Error generating TTS audio: {e}")
            return ""
            
    return filepath
