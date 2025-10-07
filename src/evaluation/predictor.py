import numpy as np
from tqdm import tqdm
from config.hyperparameters import config

class SentimentPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict(self, X_test):
        """Performs batch inference on the test set."""
        
        y_pred = []
        # Convert DataFrame column to a list of prompts
        prompts = X_test["text"].tolist()

        # Set batch size depending on GPU memory
        batch_size = 8
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, 
                truncation=True, max_length=config.MAX_SEQ_LENGTH
            ).to("cuda")
            
            outputs = self.model.generate(
                **inputs,
                # Set a higher max_new_tokens to ensure the model can generate full words
                max_new_tokens=10,
                do_sample=False,  # Use greedy decoding for deterministic output
                top_p=1.0,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and parse the generated text
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for output in decoded_outputs:
                # The generated answer is after the last '=' sign
                answer = output.split("=")[-1].lower().strip()
                
                if "positive" in answer:
                    y_pred.append("positive")
                elif "negative" in answer:
                    y_pred.append("negative")
                elif "neutral" in answer:
                    y_pred.append("neutral")
                else:
                    # Fallback for unexpected or empty outputs
                    y_pred.append("none")
        
        return y_pred