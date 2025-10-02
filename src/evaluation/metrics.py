from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class EvaluationMetrics:
    def __init__(self):
        """Calculates comprehensive evaluation metrics."""
        self.labels = ['positive', 'neutral', 'negative']
        self.mapping = {'positive': 2, 'neutral': 1, 'none': 1, 'negative': 0}
    
    def evaluate(self, y_true, y_pred):
        def map_func(x):
            return self.mapping.get(x, 1)
        
        y_true_mapped = np.vectorize(map_func)(y_true)
        y_pred_mapped = np.vectorize(map_func)(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_mapped, y_pred_mapped)
        class_report = classification_report(y_true_mapped, y_pred_mapped)
        conf_matrix = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1, 2])
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }