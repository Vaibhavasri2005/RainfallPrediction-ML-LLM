"""
Data Preprocessing Module
Handles loading, cleaning, and normalizing weather data for the rainfall prediction model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class WeatherDataPreprocessor:
    """Preprocesses weather data for rainfall prediction."""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_clean(self, data_path=None, df=None):
        """
        Load and clean weather data.
        
        Args:
            data_path (str): Path to CSV file
            df (pd.DataFrame): DataFrame if data already loaded
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df is None:
            df = pd.read_csv(data_path)
        
        # Remove missing values
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Ensure required columns exist
        required_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def preprocess(self, df):
        """
        Preprocess data: clean, split, and normalize.
        
        Args:
            df (pd.DataFrame): Raw weather data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Clean data
        df = self.load_and_clean(df=df)
        
        # Extract features and target
        features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        self.feature_names = features
        
        X = df[features].values
        # Create binary target: 1 if rainfall > 0, 0 otherwise
        y = (df['rainfall'] > 0).astype(int).values
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Normalize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def transform_single_instance(self, instance):
        """
        Transform a single weather instance for prediction.
        
        Args:
            instance (dict or array): Weather data with keys/order matching features
            
        Returns:
            np.array: Normalized feature vector
        """
        if isinstance(instance, dict):
            values = np.array([instance.get(f) for f in self.feature_names]).reshape(1, -1)
        else:
            values = np.array(instance).reshape(1, -1)
        
        return self.scaler.transform(values)
    
    def get_feature_names(self):
        """Get the names of features used."""
        return self.feature_names
