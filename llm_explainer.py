"""
LLM-Based Explanation Module
Converts model predictions into human-readable insights.
"""

import numpy as np


class RainfallExplainer:
    """
    LLM-style explanation module that converts predictions into human-readable insights.
    This simulates an LLM by generating contextual explanations based on feature importance
    and model predictions.
    """
    
    def __init__(self):
        """Initialize the explainer."""
        self.explanation_templates = self._load_templates()
    
    def _load_templates(self):
        """Load explanation templates for different scenarios."""
        return {
            'rain_high_confidence': [
                "Strong indicators suggest rain is likely.",
                "Multiple weather factors point towards rainfall.",
                "Atmospheric conditions are favorable for precipitation."
            ],
            'rain_medium_confidence': [
                "Weather conditions suggest rain is possible.",
                "Some indicators point towards possible rainfall.",
                "There's a moderate chance of rain."
            ],
            'rain_low_confidence': [
                "Light rain might occur, but conditions are mixed.",
                "Rain is unlikely but cannot be ruled out.",
                "Limited indicators suggest precipitation."
            ],
            'no_rain_high_confidence': [
                "Current conditions indicate no rain is expected.",
                "Dry weather patterns are evident.",
                "Atmospheric conditions favor clear skies."
            ],
            'no_rain_medium_confidence': [
                "Rain is unlikely, but conditions are borderline.",
                "Clear weather is expected with some uncertainty.",
                "Most indicators suggest no precipitation."
            ],
            'no_rain_low_confidence': [
                "Conditions are mixed; rain is not expected but possible.",
                "The forecast is uncertain; clear weather is slightly more likely.",
                "Weak indicators suggest no rain, but watch for changes."
            ]
        }
    
    def explain_prediction(self, prediction_details, weather_data):
        """
        Generate a comprehensive explanation for a prediction.
        
        Args:
            prediction_details (dict): Output from model.get_prediction_details()
            weather_data (dict): Raw weather values with feature names as keys
            
        Returns:
            dict: Explanation containing main insight, confidence, factors, and full text
        """
        prediction = prediction_details['prediction']
        prob_rain = prediction_details['probability_rain']
        prob_no_rain = prediction_details['probability_no_rain']
        feature_contributions = prediction_details['feature_contributions']
        feature_importance = prediction_details['feature_importance']
        
        # Determine confidence level
        max_prob = max(prob_rain, prob_no_rain)
        confidence = self._get_confidence_level(max_prob)
        
        # Determine which factors are most influential
        top_factors = self._get_top_factors(feature_contributions, feature_importance, weather_data)
        
        # Generate base explanation
        if prediction == 1:  # Rain predicted
            base_explanation = self._select_template('rain', confidence)
        else:  # No rain predicted
            base_explanation = self._select_template('no_rain', confidence)
        
        # Build detailed factors explanation
        factors_explanation = self._build_factors_explanation(top_factors, prediction)
        
        # Build full explanation text
        full_explanation = f"{base_explanation}\n\n{factors_explanation}"
        
        return {
            'prediction': 'Rain Expected' if prediction == 1 else 'No Rain Expected',
            'confidence': confidence,
            'confidence_percentage': max_prob * 100,
            'probability_rain': prob_rain * 100,
            'probability_no_rain': prob_no_rain * 100,
            'main_insight': base_explanation,
            'factors': top_factors,
            'full_explanation': full_explanation
        }
    
    def _select_template(self, rain_status, confidence):
        """Select appropriate explanation template."""
        if rain_status == 'rain':
            if confidence == 'high':
                category = 'rain_high_confidence'
            elif confidence == 'medium':
                category = 'rain_medium_confidence'
            else:
                category = 'rain_low_confidence'
        else:
            if confidence == 'high':
                category = 'no_rain_high_confidence'
            elif confidence == 'medium':
                category = 'no_rain_medium_confidence'
            else:
                category = 'no_rain_low_confidence'
        
        templates = self.explanation_templates[category]
        return templates[hash(category) % len(templates)]
    
    def _get_confidence_level(self, probability):
        """
        Determine confidence level from probability.
        
        Args:
            probability (float): Predicted probability
            
        Returns:
            str: 'high', 'medium', or 'low'
        """
        if probability >= 0.75:
            return 'high'
        elif probability >= 0.55:
            return 'medium'
        else:
            return 'low'
    
    def _get_top_factors(self, contributions, importance, weather_data, top_n=3):
        """
        Identify top contributing factors.
        
        Args:
            contributions (dict): Feature contributions to prediction
            importance (dict): Feature importance scores
            weather_data (dict): Raw weather values
            top_n (int): Number of top factors to return
            
        Returns:
            list: Top factors with their descriptions
        """
        # Sort by absolute contribution
        sorted_factors = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        factors = []
        for feature_name, contribution in sorted_factors:
            direction = 'increase' if contribution > 0 else 'decrease'
            value = weather_data.get(feature_name, 'N/A')
            
            factors.append({
                'feature': feature_name,
                'value': value,
                'contribution': contribution,
                'direction': direction,
                'importance': importance.get(feature_name, 0)
            })
        
        return factors
    
    def _build_factors_explanation(self, top_factors, prediction):
        """
        Build a detailed explanation based on top factors.
        
        Args:
            top_factors (list): List of top contributing factors
            prediction (int): 0 for no rain, 1 for rain
            
        Returns:
            str: Explanation of contributing factors
        """
        if not top_factors:
            return "Unable to identify specific contributing factors."
        
        lines = ["Key contributing factors:"]
        
        for factor in top_factors:
            feature = factor['feature'].replace('_', ' ').title()
            value = factor['value']
            direction = factor['direction']
            contribution = abs(factor['contribution'])
            
            # Create contextual description
            if feature == 'Humidity':
                if prediction == 1 and direction == 'increase':
                    effect = "Higher humidity increases the likelihood of rain"
                elif prediction == 1 and direction == 'decrease':
                    effect = "Lower humidity reduces rain probability"
                else:
                    effect = f"{feature} is {direction}ing the prediction"
            elif feature == 'Pressure':
                if prediction == 1 and direction == 'decrease':
                    effect = "Lower atmospheric pressure indicates rain formation"
                else:
                    effect = f"{feature} is {direction}ing the prediction"
            elif feature == 'Temperature':
                effect = f"{feature} (currently {value}°C) is {direction}ing the prediction"
            elif feature == 'Wind Speed':
                effect = f"{feature} is {direction}ing the prediction"
            else:
                effect = f"{feature} is {direction}ing the prediction"
            
            lines.append(f"  • {effect} ({contribution:.3f} contribution)")
        
        return '\n'.join(lines)
    
    def print_explanation(self, explanation):
        """
        Pretty print an explanation.
        
        Args:
            explanation (dict): Explanation dictionary
        """
        print("\n" + "="*60)
        print("RAINFALL PREDICTION EXPLANATION")
        print("="*60)
        print(f"\nPrediction: {explanation['prediction']}")
        print(f"Confidence: {explanation['confidence'].upper()} ({explanation['confidence_percentage']:.1f}%)")
        print(f"Rain Probability: {explanation['probability_rain']:.1f}%")
        print(f"No Rain Probability: {explanation['probability_no_rain']:.1f}%")
        print(f"\n{explanation['main_insight']}")
        print(f"\n{explanation['factors']}")
        print("\n" + explanation['full_explanation'])
        print("="*60 + "\n")
