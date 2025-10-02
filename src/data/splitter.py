import pandas as pd
from sklearn.model_selection import train_test_split
from config.hyperparameters import config

class DataSplitter:
    def stratified_split(self, df):
        # Stratified sampling to create balanced train and test sets
        X_train, X_test = [], []
        
        for sentiment in ["positive", "neutral", "negative"]:
            sentiment_data = df[df.sentiment == sentiment]
            if len(sentiment_data) < (config.TRAIN_SAMPLES_PER_CLASS + config.TEST_SAMPLES_PER_CLASS):
                raise ValueError(f"Not enough samples for {sentiment} sentiment")
                
            train, test = train_test_split(
                sentiment_data,
                train_size=config.TRAIN_SAMPLES_PER_CLASS,
                test_size=config.TEST_SAMPLES_PER_CLASS,
                random_state=config.SEED
            )
            X_train.append(train)
            X_test.append(test)
        
        # Concatenate and shuffle the training data
        X_train = pd.concat(X_train).sample(frac=1, random_state=config.SEED)
        X_test = pd.concat(X_test)
        
        # Create a balanced evaluation set from the remaining data
        eval_index = [index for index in df.index if index not in 
                   list(X_train.index) + list(X_test.index)]
        X_eval = df[df.index.isin(eval_index)]
        X_eval = (X_eval.groupby('sentiment', group_keys=False)
                  .apply(lambda x: x.sample(n=config.EVAL_SAMPLES_PER_CLASS, 
                                          random_state=config.SEED, 
                                          replace=True)))
        
        return X_train, X_test, X_eval