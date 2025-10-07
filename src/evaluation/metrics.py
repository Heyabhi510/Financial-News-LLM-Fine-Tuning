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

        # Generate accuracy report
        unique_labels = set(y_true_mapped)  # Get unique labels

        for label in unique_labels:
            label_indices = [i for i in range(len(y_true_mapped))
                            if y_true_mapped[i] == label]
            label_y_true_mapped = [y_true_mapped[i] for i in label_indices]
            label_y_pred_mapped = [y_pred_mapped[i] for i in label_indices]
            accuracy = accuracy_score(label_y_true_mapped, label_y_pred_mapped)
            print(f'Accuracy for label {label}: {accuracy:.3f}')
        
        # Calculate metrics
        accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
        class_report = classification_report(y_true_mapped, y_pred_mapped)
        conf_matrix = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1, 2])
        
        return {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }