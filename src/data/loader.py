import pandas as pd
from sklearn.model_selection import train_test_split
from config.paths import paths
from config.hyperparameters import config

class DataLoader:
    def __init__(self):
        self.df = None
    
    def load_raw_data(self):
        """Load and basic validation of raw data"""
        self.df = pd.read_csv(
            paths.RAW_DATA, 
            names=["sentiment", "text"],
            encoding="utf-8", 
            encoding_errors="replace"
        )
        print(f"Loaded {len(self.df)} samples")
        return self.df
    
    def validate_data(self):
        """Basic data validation"""
        assert not self.df.empty, "Dataframe is empty"
        assert 'sentiment' in self.df.columns, "Missing sentiment column"
        assert 'text' in self.df.columns, "Missing text column"
        print("Data validation passed")
