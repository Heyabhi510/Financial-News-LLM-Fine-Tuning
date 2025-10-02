#!/usr/bin/env python3
"""
Main training script for Gemma Financial Sentiment Analysis
"""

import sys
sys.path.append('/content/Financial-News-LLM-Fine-Tuning')

from config.paths import paths
from config.hyperparameters import config
from data.loader import DataLoader
from data.splitter import DataSplitter
from data.preprocessor import PromptEngineer
from models.setup import ModelSetup
from training.train import ModelTrainer
from evaluation.predictor import SentimentPredictor
from evaluation.metrics import EvaluationMetrics
from utils.logger import setup_logging

def main():
    # Setup
    paths.setup()
    logger = setup_logging()
    
    # Hugging Face token
    hf_token = "HF_Token"
    
    try:
        logger.info("Starting Gemma Fine-tuning Pipeline")
        
        # 1. Load data
        logger.info("Loading data...")
        data_loader = DataLoader()
        df = data_loader.load_raw_data()
        data_loader.validate_data()
        
        # 2. Split data
        logger.info("Splitting data...")
        splitter = DataSplitter()
        X_train, X_test, X_eval = splitter.stratified_split(df)
        
        # 3. Setup model
        logger.info("Setting up model and tokenizer...")
        model_setup = ModelSetup(hf_token)
        model = model_setup.setup_model()
        tokenizer = model_setup.setup_tokenizer()
        
        # 4. Prepare prompts and datasets
        logger.info("Preparing prompts...")
        prompt_engineer = PromptEngineer(tokenizer)
        train_data, eval_data, X_test_prompt, y_true = prompt_engineer.prepare_datasets(
            X_train, X_test, X_eval
        )
        
        # 5. Train model
        logger.info("Starting training...")
        trainer = ModelTrainer(model, tokenizer)
        sft_trainer = trainer.create_trainer(train_data, eval_data)
        sft_trainer.train()
        
        # 6. Save model
        logger.info("Saving model...")
        sft_trainer.model.save_pretrained(str(paths.TRAINED_MODEL))
        
        # 7. Evaluate model
        logger.info("Evaluating model...")
        predictor = SentimentPredictor(model, tokenizer)
        y_pred = predictor.predict(X_test_prompt)
        
        evaluator = EvaluationMetrics()
        results = evaluator.evaluate(y_true, y_pred)
        
        logger.info(f"Final Accuracy: {results['accuracy']:.3f}")
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()