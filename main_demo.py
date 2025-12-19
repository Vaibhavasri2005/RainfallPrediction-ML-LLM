"""
Main Demo Script
Demonstrates the complete rainfall prediction pipeline with ML and LLM explanation.
"""

from data_preprocessing import WeatherDataPreprocessor
from ml_model import RainfallPredictor
from llm_explainer import RainfallExplainer
from model_evaluation import ModelEvaluator
from dataset_generator import generate_sample_dataset
import numpy as np


def main():
    """Run the complete rainfall prediction pipeline."""
    
    print("\n" + "="*70)
    print("RAINFALL PREDICTION PROJECT - ML + LLM EXPLANATION MODULE")
    print("="*70)
    
    # ============================================================================
    # Step 1: Generate and Preprocess Data
    # ============================================================================
    print("\n[STEP 1] Generating and Preprocessing Weather Data...")
    print("-" * 70)
    
    # Generate synthetic dataset
    df = generate_sample_dataset(n_samples=200, random_state=42)
    print(f"✓ Generated dataset with {len(df)} weather records")
    print(f"  Rain cases: {(df['rainfall'] > 0).sum()}")
    print(f"  No-rain cases: {(df['rainfall'] == 0).sum()}")
    
    # Preprocess data
    preprocessor = WeatherDataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    print(f"\n✓ Data preprocessing complete:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {', '.join(preprocessor.get_feature_names())}")
    
    # ============================================================================
    # Step 2: Train ML Model
    # ============================================================================
    print("\n[STEP 2] Training Logistic Regression Model...")
    print("-" * 70)
    
    model = RainfallPredictor(random_state=42)
    model.train(X_train, y_train, feature_names=preprocessor.get_feature_names())
    
    # ============================================================================
    # Step 3: Evaluate Model
    # ============================================================================
    print("\n[STEP 3] Evaluating Model Performance...")
    print("-" * 70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)
    evaluator.print_report()
    
    # ============================================================================
    # Step 4: Make Predictions with LLM-Style Explanations
    # ============================================================================
    print("\n[STEP 4] Making Predictions with LLM Explanations...")
    print("-" * 70)
    
    explainer = RainfallExplainer()
    
    # Example predictions on new data
    print("\nExample Predictions:\n")
    
    # Test case 1: High humidity, low pressure (likely rain)
    test_case_1 = np.array([[0.5, 2.0, -1.5, -0.5]])  # Normalized values
    raw_weather_1 = {'temperature': 25, 'humidity': 85, 'pressure': 1005, 'wind_speed': 12}
    
    print("TEST CASE 1: High humidity and low pressure")
    print("-" * 50)
    prediction_details_1 = model.get_prediction_details(test_case_1)
    explanation_1 = explainer.explain_prediction(prediction_details_1, raw_weather_1)
    explainer.print_explanation(explanation_1)
    
    # Test case 2: Low humidity, high pressure (likely no rain)
    test_case_2 = np.array([[-1.5, -1.8, 1.2, 0.3]])  # Normalized values
    raw_weather_2 = {'temperature': 18, 'humidity': 35, 'pressure': 1025, 'wind_speed': 5}
    
    print("\nTEST CASE 2: Low humidity and high pressure")
    print("-" * 50)
    prediction_details_2 = model.get_prediction_details(test_case_2)
    explanation_2 = explainer.explain_prediction(prediction_details_2, raw_weather_2)
    explainer.print_explanation(explanation_2)
    
    # Test case 3: Moderate conditions (borderline case)
    test_case_3 = np.array([[0.1, 0.3, -0.2, 0.4]])  # Normalized values
    raw_weather_3 = {'temperature': 22, 'humidity': 60, 'pressure': 1012, 'wind_speed': 8}
    
    print("\nTEST CASE 3: Moderate weather conditions")
    print("-" * 50)
    prediction_details_3 = model.get_prediction_details(test_case_3)
    explanation_3 = explainer.explain_prediction(prediction_details_3, raw_weather_3)
    explainer.print_explanation(explanation_3)
    
    # ============================================================================
    # Step 5: Detailed Analysis Report
    # ============================================================================
    print("\n[STEP 5] Detailed Analysis and Statistics...")
    print("-" * 70)
    
    print("\nMetrics Summary:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric:20s}: {value:.4f}")
        else:
            print(f"  {metric:20s}: {value}")
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"""
Project Overview:
1. Data Preprocessing: Converted raw weather data into normalized features
2. Model Training: Trained Logistic Regression on {X_train.shape[0]} samples
3. Model Evaluation: Achieved {metrics['accuracy']:.2%} accuracy on test set
4. LLM Explanations: Generated human-readable predictions for new weather data

Key Insights:
  • Model Accuracy: {metrics['accuracy']:.2%}
  • Precision (rain prediction): {metrics['precision']:.2%}
  • Recall (detecting actual rain): {metrics['recall']:.2%}
  • F1 Score: {metrics['f1_score']:.4f}
  
The system successfully combines machine learning predictions with 
AI-style explanations to make complex weather patterns understandable to users.
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
