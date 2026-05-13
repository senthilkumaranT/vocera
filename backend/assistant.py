import os
import sys
import torch
import torchaudio
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import from the services directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.stt_services import STTService
from services.llm_services import LLMService
from services.tts_services import TTSService

def record_audio(filename="user_input.wav", duration=5, fs=16000):
    """
    Records audio from the default microphone for a fixed duration.
    """
    print(f"\n🎙️  [Microphone] Recording for {duration} seconds... Please speak now!")
    # Record audio (mono)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ [Microphone] Recording finished.")
    
    # Save using torchaudio
    waveform = torch.from_numpy(recording).T # Shape: [channels, time]
    
    # Ensure sampledata directory exists
    os.makedirs("sampledata", exist_ok=True)
    filepath = os.path.join("sampledata", filename)
    torchaudio.save(filepath, waveform, fs)
    return filepath

def play_audio(filename):
    """
    Plays the given audio file using sounddevice.
    """
    print(f"🔊 [Speaker] Playing response...")
    waveform, sample_rate = torchaudio.load(filename)
    audio_data = waveform.numpy().T
    sd.play(audio_data, sample_rate)
    sd.wait()
    print("✅ [Speaker] Finished playing.")

def main():
    print("="*60)
    print("🚀 Initializing Vocera Voice Assistant...")
    print("This will load STT, LLM, and TTS models. Please wait.")
    print("="*60)
    
    stt = STTService()
    llm = LLMService()
    tts = TTSService()
    
    print("\n" + "="*60)
    print("✨ Assistant is ready! ✨")
    print("="*60)
    
    while True:
        try:
            input("\nPress ENTER to start recording your question (or Ctrl+C to exit)...")
            
            # 1. Record audio (STT Input)
            input_audio_path = record_audio(duration=6) # 6 seconds recording time
            
            # 2. Transcribe (STT)
            print("\n📝 Transcribing...")
            user_text = stt.transcribe(input_audio_path)
            print(f"🗣️  You said: '{user_text}'")
            
            if not user_text.strip():
                print("⚠️  Could not hear anything. Please try again.")
                continue
                
            # 3. Generate Answer (LLM)
            print("\n🧠 Generating answer...")
            answer = llm.generate_response(user_text)
            print(f"🤖 Assistant: '{answer}'")
            
            if not answer.strip():
                print("⚠️  Empty response from LLM.")
                continue
            
            # 4. Synthesize Speech (TTS)
            print("\n🎵 Converting answer to speech...")
            output_audio_path = tts.synthesize(answer)
            
            # 5. Play Answer
            play_audio(output_audio_path)
            
        except KeyboardInterrupt:
            print("\n\n👋 Exiting Vocera Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
