from transformers import VitsModel, AutoTokenizer
import torch
import torchaudio
import os

def test_tts():
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    text = "Hello, this is a test of the text to speech service."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    torchaudio.save("test_tts.wav", output, model.config.sampling_rate)
    print("TTS test successful.")

if __name__ == "__main__":
    test_tts()
