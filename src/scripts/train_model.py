#!/usr/bin/env python3
"""
Main training script for Gemma Financial Sentiment Analysis
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
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
    hf_token = os.environ.get("HF_TOKEN")
    
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
        tokenizer, model = model_setup.setup_tokenizer(model)
        
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

        # Access the log history
        log_history = sft_trainer.state.log_history

        # Extract training / validation loss
        train_losses = [log["loss"] for log in log_history if "loss" in log]
        epoch_train = [log["epoch"] for log in log_history if "loss" in log]
        eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
        epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

        # Plot the training loss
        plt.plot(epoch_train, train_losses, label="Training Loss")
        plt.plot(epoch_eval, eval_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()

        # tensorboard
        print(f"tensorboard --logdir {paths.TRAINED_MODEL}/runs")
        
        # 7. Evaluate model
        predictor = SentimentPredictor(model, tokenizer)
        y_pred = predictor.predict(X_test_prompt)
        
        evaluator = EvaluationMetrics()
        results = evaluator.evaluate(y_true, y_pred)

        # Set model configuration for inference
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

        y_pred = predictor.predict(X_test)
        evaluator.evaluate(y_true, y_pred)

        evaluation_df = pd.DataFrame({'text': X_test["text"], 'y_true':y_true, 'y_pred': y_pred})
        evaluation_df.to_csv("test_predictions.csv", index=False)

        print("Predictions saved to test_predictions.csv")
        evaluation_df.head()
        
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()