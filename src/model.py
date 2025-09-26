import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

class F1RacePredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.7, 
            max_depth=3, 
            random_state=37
        )
        self.imputer = SimpleImputer(strategy="median")
        self.feature_columns = [
            "QualifyingTime", "RainProbability", "Temperature", 
            "TeamPerformanceScore", "CleanAirRacePace (s)", "AveragePositionChange"
        ]
        
    def prepare_data(self, qualifying_data, sector_times, laps_data, rain_probability, temperature):
        """Prepare data for model training"""
        # Merge data
        merged_data = qualifying_data.merge(
            sector_times[["Driver", "TotalSectorTime (s)"]], 
            on="Driver", 
            how="left"
        )
        merged_data["RainProbability"] = rain_probability
        merged_data["Temperature"] = temperature
        
        # Filter valid drivers
        valid_drivers = merged_data["Driver"].isin(laps_data["Driver"].unique())
        merged_data = merged_data[valid_drivers]
        
        # Prepare features and target
        X = merged_data[self.feature_columns]
        y = laps_data.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
        
        return merged_data, X, y
    
    def train_and_predict(self, X, y):
        """Train model and make predictions"""
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Check if we have enough data for train-test split
        if len(X_imputed) <= 1:
            # If not enough data, fit on all data and return predictions without calculating MAE
            self.model.fit(X_imputed, y)
            predictions = self.model.predict(X_imputed)
            return predictions, float('inf')  # Return infinity as MAE when not enough data for validation
        
        # Train-test split if we have enough data
        if len(X_imputed) < 5:  # If fewer than 5 samples, use simpler validation
            test_size = max(1, int(0.2 * len(X_imputed)))  # Use 20% or at least 1 for test
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=test_size, random_state=37
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=0.3, random_state=37
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate error if we have test data
        if len(X_test) > 0 and len(y_test) > 0:
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
        else:
            mae = float('inf')  # If no test data, return infinity as MAE
        
        # Make predictions on full dataset
        predictions = self.model.predict(X_imputed)
        
        return predictions, mae
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        return dict(zip(self.feature_columns, self.model.feature_importances_))