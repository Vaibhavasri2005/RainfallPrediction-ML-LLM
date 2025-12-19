"""
Machine Learning Model Module
Contains the Logistic Regression model for rainfall prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


class RainfallPredictor:
    """Logistic Regression model for rainfall prediction."""
    
    def __init__(self, random_state=42):
        """
        Initialize the rainfall predictor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the logistic regression model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            feature_names (list): Names of features for interpretation
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Calculate feature importance from coefficients
        self.feature_importance = np.abs(self.model.coef_[0])
        
        print("✓ Model trained successfully")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Feature vectors
            
        Returns:
            np.array: Predicted classes (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Feature vectors
            
        Returns:
            np.array: Probability estimates
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_prediction_details(self, X_single):
        """
        Get detailed prediction information for a single instance.
        
        Args:
            X_single (np.array): Single feature vector (shape: 1, n_features)
            
        Returns:
            dict: Prediction details including probabilities and feature contributions
        """
        prediction = self.predict(X_single)[0]
        probabilities = self.predict_proba(X_single)[0]
        
        # Calculate feature contributions
        feature_values = X_single[0]
        feature_contributions = feature_values * self.model.coef_[0]
        
        return {
            'prediction': prediction,
            'probability_no_rain': probabilities[0],
            'probability_rain': probabilities[1],
            'feature_contributions': dict(zip(self.feature_names, feature_contributions)),
            'feature_importance': dict(zip(self.feature_names, self.feature_importance))
        }
    
    def save_model(self, filepath):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")
