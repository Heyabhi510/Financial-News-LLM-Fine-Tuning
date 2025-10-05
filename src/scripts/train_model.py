#!/usr/bin/env python3
"""
Main training script for Gemma Financial Sentiment Analysis
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from config.paths import paths
from data.loader import DataLoader
from data.splitter import DataSplitter
from data.preprocessor import PromptEngineer
from models.setup import ModelSetup
from training.train import ModelTrainer
from evaluation.predictor import SentimentPredictor
from evaluation.metrics import EvaluationMetrics

def main():
    # Setup
    paths.setup()
    
    # Hugging Face token
    hf_token = "HF_Token"
    
    try:
        
        # 1. Load data
        data_loader = DataLoader()
        df = data_loader.load_raw_data()
        data_loader.validate_data()
        
        # 2. Split data
        splitter = DataSplitter()
        X_train, X_test, X_eval = splitter.stratified_split(df)
        
        # 3. Setup model
        model_setup = ModelSetup(hf_token)
        model = model_setup.setup_model()
        tokenizer = model_setup.setup_tokenizer()
        
        # 4. Prepare prompts and datasets
        prompt_engineer = PromptEngineer(tokenizer)
        train_data, eval_data, X_test_prompt, y_true = prompt_engineer.prepare_datasets(
            X_train, X_test, X_eval
        )
        
        # 5. Train model
        trainer = ModelTrainer(model, tokenizer)
        sft_trainer = trainer.create_trainer(train_data, eval_data)
        sft_trainer.train()
        
        # 6. Save model
        sft_trainer.model.save_pretrained(str(paths.TRAINED_MODEL))
        
        # 7. Evaluate model
        predictor = SentimentPredictor(model, tokenizer)
        y_pred = predictor.predict(X_test_prompt)
        
        evaluator = EvaluationMetrics()
        results = evaluator.evaluate(y_true, y_pred)
        
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()