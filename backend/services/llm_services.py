import torch
import logging
from transformers import pipeline
from langsmith import traceable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initializes the Language Model service with a lightweight HuggingFace model.
        
        Args:
            model_name (str): The Hugging Face model identifier to use.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading LLM model '{self.model_name}' on {self.device}...")
        
        self.pipe = pipeline(
            "text-generation", 
            model=self.model_name, 
            torch_dtype=self.torch_dtype, 
            device_map="auto" if torch.cuda.is_available() else None
        )
        logger.info("LLM loaded successfully.")

    @traceable
    def generate_response(self, user_text: str) -> str:
        """
        Generates a conversational response based on user input.
        
        Args:
            user_text (str): The transcribed text from the user.
            
        Returns:
            str: The generated response from the LLM.
        """
        logger.info(f"Generating response for: '{user_text}'")
        
        messages = [
            {"role": "system", "content": "You are a helpful, brief voice assistant. Keep answers short, conversational, and to the point (1-2 sentences). Do not use emojis."},
            {"role": "user", "content": user_text},
        ]
        
        try:
            prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback format if chat template fails
            prompt = f"<|system|>\nYou are a helpful, brief voice assistant. Keep answers short, usually 1 or 2 sentences.</s>\n<|user|>\n{user_text}</s>\n<|assistant|>\n"
            
        outputs = self.pipe(
            prompt, 
            max_new_tokens=75, 
            do_sample=True, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95,
            pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        
        # Extract only the newly generated text
        generated_text = outputs[0]["generated_text"]
        
        if prompt in generated_text:
            response = generated_text[len(prompt):].strip()
        else:
            # Fallback if parsing fails
            response = generated_text.split("<|assistant|>")[-1].strip()
            
        logger.info(f"Response: '{response}'")
        return response

if __name__ == "__main__":
    # Example usage
    llm = LLMService()
    test_text = "What is the capital of France?"
    answer = llm.generate_response(test_text)
    print(f"\n[Success] Generated Answer: {answer}")
