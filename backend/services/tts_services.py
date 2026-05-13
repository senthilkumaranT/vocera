import os
import torch
import torchaudio
import logging
from datetime import datetime
from transformers import VitsModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, model_name="facebook/mms-tts-eng", output_dir="../sampledata"):
        """
        Initializes the Text-to-Speech service with a small HuggingFace model.
        
        Args:
            model_name (str): The Hugging Face model identifier to use.
            output_dir (str): Relative or absolute path where the audio files will be saved.
        """
        self.model_name = model_name
        
        # Resolve output directory relative to this script's location
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.abspath(os.path.join(base_dir, output_dir))
        
        logger.info(f"Initializing TTS Service using model '{self.model_name}'")
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Audio output directory set to: {self.output_dir}")
        
        logger.info("Loading tokenizer and model (this might take a moment if downloading)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = VitsModel.from_pretrained(self.model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded successfully on {self.device}.")

    def synthesize(self, text: str, filename: str = None) -> str:
        """
        Synthesizes speech from text and saves it to a WAV file in the sampledata folder.
        
        Args:
            text (str): The text to convert to speech.
            filename (str, optional): The name of the output file. If None, a timestamped name is generated.
            
        Returns:
            str: The absolute path to the generated audio file.
        """
        logger.info(f"Synthesizing text: '{text}'")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate speech
        with torch.no_grad():
            output = self.model(**inputs).waveform.cpu()

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_output_{timestamp}.wav"
            
        file_path = os.path.join(self.output_dir, filename)
        
        # Save the audio using torchaudio
        torchaudio.save(file_path, output, self.model.config.sampling_rate)
        logger.info(f"Audio saved successfully to: {file_path}")
        
        return file_path

if __name__ == "__main__":
    # Example usage
    tts = TTSService()
    test_text = "Welcome to Vocera. This is a demonstration of our new text to speech service."
    output_path = tts.synthesize(test_text)
    print(f"\n[Success] Synthesis completed. Check the audio file at: {output_path}")
