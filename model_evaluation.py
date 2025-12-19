"""
Model Evaluation Module
Computes various metrics to evaluate model performance.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluates rainfall prediction model performance."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
        self.confusion = None
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_pred_proba (np.array): Predicted probabilities (optional)
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        self.confusion = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = self.confusion.ravel()
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)
        
        # Additional metrics
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return self.metrics
    
    def print_report(self):
        """Print a formatted evaluation report."""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:    {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"  Precision:   {self.metrics['precision']:.4f}")
        print(f"  Recall:      {self.metrics['recall']:.4f}")
        print(f"  F1 Score:    {self.metrics['f1_score']:.4f}")
        print(f"  Specificity: {self.metrics['specificity']:.4f}")
        print(f"  Sensitivity: {self.metrics['sensitivity']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"  ROC-AUC:     {self.metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {self.metrics['true_negatives']}")
        print(f"  False Positives: {self.metrics['false_positives']}")
        print(f"  False Negatives: {self.metrics['false_negatives']}")
        print(f"  True Positives:  {self.metrics['true_positives']}")
        
        print("\nInterpretation:")
        print(f"  • Out of {self.metrics['true_negatives'] + self.metrics['false_positives']} non-rain cases,")
        print(f"    the model correctly identified {self.metrics['true_negatives']} ({self.metrics['specificity']*100:.1f}%)")
        print(f"  • Out of {self.metrics['true_positives'] + self.metrics['false_negatives']} rain cases,")
        print(f"    the model correctly identified {self.metrics['true_positives']} ({self.metrics['sensitivity']*100:.1f}%)")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            save_path (str): Path to save figure (optional)
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Rain', 'Rain'],
                    yticklabels=['No Rain', 'Rain'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Rainfall Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path=None):
        """
        Plot metrics comparison bar chart.
        
        Args:
            save_path (str): Path to save figure (optional)
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']
        values = [self.metrics.get(m, 0) for m in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_to_plot, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim([0, 1.1])
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Metrics chart saved to {save_path}")
        
        plt.show()
    
    def get_metrics_dict(self):
        """Return metrics as dictionary."""
        return self.metrics.copy()
