from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score
import numpy as np
import logging

class Evaluator:
    @staticmethod
    def _protected_div(a, b):
        """Protected division to avoid NaN/inf"""
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(a, b)
            return np.where(np.abs(b) > 1e-6, res, 1.0)

    @staticmethod
    def evaluate_regression(model, X, y):
        """Robust regression evaluation with logging"""
        try:
            y_pred = model.predict(X)
            if len(np.unique(y_pred)) < 5:
                logging.warning(f"[Evaluator] Model output has low variance. Preds: {np.unique(y_pred)}")
                return {'mse': np.inf, 'r2': -np.inf, 'complexity': np.inf}

            mse = mean_squared_error(y, y_pred)
            r2 = max(-1.0, r2_score(y, y_pred))
            complexity = getattr(model, 'complexity_', len(model.pareto_front[0]))

            print(f"[Evaluator - Regression] MSE: {mse:.4f}, Square(R): {r2:.4f}, Complexity: {complexity}")
            return {'mse': mse, 'r2': r2, 'complexity': complexity}
        except Exception as e:
            logging.error(f"[Evaluator] Regression evaluation failed: {e}")
            return {'mse': np.inf, 'r2': -np.inf, 'complexity': np.inf}

    @staticmethod
    def evaluate_classification(model, X, y):
        """Robust classification evaluation with logging"""
        try:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            complexity = getattr(model, 'complexity_', len(model.pareto_front[0]))

            print(f"[Evaluator - Classification] Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Complexity: {complexity}")
            return {
                'accuracy': acc,
                'precision': prec,
                'recall': recall,
                'f1': f1,
                'complexity': complexity
            }
        except Exception as e:
            logging.error(f"[Evaluator] Classification evaluation failed: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'complexity': np.inf}
