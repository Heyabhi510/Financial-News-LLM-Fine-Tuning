from datasets import Dataset
import pandas as pd

class PromptEngineer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token
    
    def create_training_prompt(self, data_point):
        return f"""Analyze the sentiment of the news headline enclosed in square brackets, determine if it is positive, neutral, or negative, and return the answer as the corresponding sentiment label "positive" or "neutral" or "negative"

        [{data_point["text"]}] = {data_point["sentiment"]}""".strip() + self.eos_token

    def create_test_prompt(self, data_point):
        return f"""Analyze the sentiment of the news headline enclosed in square brackets, determine if it is positive, neutral, or negative, and return the answer as the corresponding sentiment label "positive" or "neutral" or "negative"

        [{data_point["text"]}] = """.strip()
    
    def prepare_datasets(self, X_train, X_test, X_eval):
        # Apply prompt formatting
        X_train["text"] = X_train.apply(self.create_training_prompt, axis=1)
        X_eval["text"] = X_eval.apply(self.create_training_prompt, axis=1)
        
        # Store true labels for final evaluation and format test set for inference
        y_true = X_test.sentiment
        X_test_prompt = pd.DataFrame(X_test.apply(self.create_test_prompt, axis=1), 
                                   columns=["text"])
        
        # Convert pandas DataFrames to Hugging Face Dataset objects
        train_data = Dataset.from_pandas(X_train)
        eval_data = Dataset.from_pandas(X_eval)
        
        return train_data, eval_data, X_test_prompt, y_true