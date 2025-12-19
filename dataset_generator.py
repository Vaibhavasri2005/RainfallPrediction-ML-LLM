"""
Sample Dataset Generator
Creates synthetic weather data for demonstration and testing.
"""

import numpy as np
import pandas as pd


def generate_sample_dataset(n_samples=200, random_state=42):
    """
    Generate synthetic weather data for rainfall prediction.
    
    Args:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with weather features and rainfall target
    """
    np.random.seed(random_state)
    
    # Generate base features
    temperature = np.random.uniform(10, 35, n_samples)  # 10-35°C
    humidity = np.random.uniform(20, 100, n_samples)    # 20-100%
    pressure = np.random.uniform(995, 1035, n_samples)  # 995-1035 hPa
    wind_speed = np.random.uniform(0, 25, n_samples)    # 0-25 km/h
    
    # Generate rainfall based on feature relationships (simulating real patterns)
    rainfall = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Higher humidity increases rainfall probability
        rain_prob = humidity[i] / 100 * 0.5
        
        # Lower pressure increases rainfall probability
        rain_prob += (1 - (pressure[i] - 995) / 40) * 0.3
        
        # Higher wind speed can increase rainfall
        rain_prob += (wind_speed[i] / 25) * 0.2
        
        # Temperature effect (moderate temps favor rain)
        temp_effect = 1 - abs(temperature[i] - 22) / 22 * 0.5
        rain_prob += temp_effect * 0.2
        
        # Add some randomness
        rain_prob += np.random.normal(0, 0.1)
        rain_prob = np.clip(rain_prob, 0, 1)
        
        # Generate rainfall amount (0 for no rain, positive for rain)
        if np.random.random() < rain_prob:
            rainfall[i] = np.random.uniform(0.1, 50)
        else:
            rainfall[i] = 0
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'rainfall': rainfall
    })
    
    return df


def save_sample_dataset(filepath, df):
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"✓ Dataset saved to {filepath}")


if __name__ == "__main__":
    # Generate and save sample dataset
    df = generate_sample_dataset(n_samples=200)
    print("\nGenerated Sample Dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"Samples with rain: {(df['rainfall'] > 0).sum()}")
    print(f"Samples without rain: {(df['rainfall'] == 0).sum()}")
