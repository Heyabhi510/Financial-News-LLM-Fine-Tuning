from pathlib import Path

class Paths:
    ROOT = Path("/content/Financial-News-LLM-Fine-Tuning/src")
    DATA = ROOT / "data"
    MODELS = ROOT / "models"
    OUTPUTS = ROOT / "outputs"
    
    # File paths
    RAW_DATA = DATA / "all-data.csv"
    TRAINED_MODEL = OUTPUTS / "trained_models" / "financial_sentiment_lora"
    PREDICTIONS = OUTPUTS / "predictions" / "test_predictions.csv"
    
    def setup(self):
        """Create necessary directories"""
        self.DATA.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS.mkdir(parents=True, exist_ok=True)
        (self.OUTPUTS / "trained_models").mkdir(exist_ok=True)
        (self.OUTPUTS / "predictions").mkdir(exist_ok=True)

paths = Paths()