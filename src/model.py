import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
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
            "TeamPerformanceScore", "CleanAirRacePace (s)", "AveragePositionChange",
            "TrackDownforce", "PitLossTime", "TotalSectorTime (s)"
        ]
        
    def prepare_data(self, qualifying_data: pd.DataFrame, sector_times: pd.DataFrame, laps_data: pd.DataFrame, results_data: pd.DataFrame, rain_probability: float, temperature: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepare data for model training"""
        # Merge data
        merged_data = qualifying_data.merge(
            sector_times[["Driver", "TotalSectorTime (s)"]], 
            on="Driver", 
            how="left"
        )
        merged_data["RainProbability"] = rain_probability
        merged_data["Temperature"] = temperature
        
        # Merge with Results to get target Position
        # results_data has ["Driver", "Position"]
        merged_data = merged_data.merge(results_data, on="Driver", how="inner")
        
        # Filter valid drivers (those who have both Quali and Result data)
        valid_drivers = merged_data["Driver"].notna()
        merged_data = merged_data[valid_drivers]
        
        # Prepare features and target
        X = merged_data[self.feature_columns]
        y = merged_data["Position"]
        
        return merged_data, X, y
    
    def train_and_predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def monte_carlo_predict(self, X: pd.DataFrame, drivers: pd.Series, n_simulations: int = 100) -> pd.DataFrame:
        """
        Run Monte Carlo simulation to estimate probabilities.
        Returns a DataFrame with Win Probability, Podium Probability, and Position stats.
        """
        # Pre-transform X to avoid repeated imputation if possible, but imputer is fast
        sim_results = []
        
        pace_idx = self.feature_columns.index("CleanAirRacePace (s)")
        pit_idx = self.feature_columns.index("PitLossTime")
        
        X_base = self.imputer.transform(X)
        
        for _ in range(n_simulations):
            X_sim = X_base.copy()
            
            # Add Pace Noise (Driver consistency/errors): 0.15s SD
            pace_noise = np.random.normal(0, 0.15, size=X_sim.shape[0])
            X_sim[:, pace_idx] += pace_noise
            
            # Add Pit Stop Noise (Crew errors): 0.5s SD
            pit_noise = np.random.normal(0, 0.5, size=X_sim.shape[0])
            X_sim[:, pit_idx] += pit_noise
            
            preds = self.model.predict(X_sim)
            sim_results.append(preds)
            
        sim_array = np.array(sim_results)
        
        stats = []
        for i, driver in enumerate(drivers):
            driver_preds = sim_array[:, i]
            
            # Win: Position approx 1.0 (lowest score wins)

        # Rank the predictions for each simulation to get integer positions (1st, 2nd...)
        # argsort twice gives ranks (0-based)
        ranks = np.argsort(np.argsort(sim_array, axis=1), axis=1) + 1
        
        stats_data = []
        for i, driver in enumerate(drivers):
            driver_ranks = ranks[:, i]
            
            win_prob = np.mean(driver_ranks == 1)
            podium_prob = np.mean(driver_ranks <= 3)
            avg_pos = np.mean(driver_ranks)
            p5_pos = np.percentile(driver_ranks, 5)
            p95_pos = np.percentile(driver_ranks, 95)
            
            stats_data.append({
                "Driver": driver,
                "Win Probability": win_prob,
                "Podium Probability": podium_prob,
                "Avg Finish": avg_pos,
                "Best Case (P5)": p5_pos,
                "Worst Case (P95)": p95_pos
            })
            
        return pd.DataFrame(stats_data).sort_values(
            by=["Win Probability", "Avg Finish"], 
            ascending=[False, True]
        ).reset_index(drop=True)