import os
import torch
import torchaudio
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STTService:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initializes the Speech-to-Text service with a HuggingFace Whisper model.
        
        Args:
            model_name (str): The Hugging Face model identifier to use.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 for CUDA to save memory and improve speed
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading STT model '{self.model_name}' on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        logger.info(f"Model loaded successfully.")

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribes speech from a WAV file to text.
        
        Args:
            audio_file_path (str): The path to the audio file.
            
        Returns:
            str: The transcribed text.
        """
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Whisper expects 16000Hz sample rate. Resample if necessary.
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Whisper processor expects a 1D numpy array
        audio_array = waveform.squeeze().numpy()
        
        inputs = self.processor(
            audio_array, 
            sampling_rate=target_sample_rate, 
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(input_features=input_features)
            
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up transcription formatting
        transcription = transcription.strip()
        
        logger.info("Transcription successful.")
        return transcription

if __name__ == "__main__":
    # Example usage
    stt = STTService()
    # Assuming test_tts.wav is in the parent directory when running from services folder
    test_audio = "../test_tts.wav"
    if os.path.exists(test_audio):
        text = stt.transcribe(test_audio)
        print(f"\n[Success] Transcription completed.\nText: {text}")
    else:
        print(f"Could not find {test_audio} for testing.")
