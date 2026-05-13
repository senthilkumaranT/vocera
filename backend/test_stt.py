import os
from services.stt_services import STTService

def test_stt():
    """
    Test script that initializes the STTService and transcribes `test_tts.wav`.
    """
    audio_file = "test_tts.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found in the current directory.")
        print("Please run `python test_tts.py` first to generate the audio file.")
        return

    print("Initializing STT Service...")
    stt_service = STTService(model_name="openai/whisper-small")
    
    print(f"\n==================================================")
    print(f"Transcribing file: {audio_file}")
    
    try:
        text = stt_service.transcribe(audio_file)
        
        print(f"==================================================")
        print(f"TRANSCRIPTION RESULT:")
        print(f"File: {audio_file}")
        print(f"Text: '{text}'")
        print(f"==================================================")
    except Exception as e:
        print(f"\nError during transcription: {e}")

if __name__ == "__main__":
    test_stt()
