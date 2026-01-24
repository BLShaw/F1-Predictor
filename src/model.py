"""
Advanced F1 Race Predictor Module.
Provides probability-based race predictions using:
1. Monte Carlo simulation (AdvancedRacePredictor)
2. Gradient Boosting ML model with SHAP explainability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN_GB = True
except ImportError:
    HAS_SKLEARN_GB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# For backwards compatibility
HAS_XGBOOST = HAS_SKLEARN_GB


class AdvancedRacePredictor:
    """
    Advanced predictor using Monte Carlo simulation to estimate race outcomes.
    Factors in:
    - Qualifying Position (Grid)
    - Race Pace (from Practice)
    - Reliability (DNF Chance)
    - Performance Variability
    """
    
    def __init__(self):
        # Base weights for score calculation
        self.grid_weight = 0.4
        self.pace_weight = 0.6
        
        # Simulation parameters
        self.base_dnf_prob = 0.05  # 5% base chance of DNF per driver
        self.pace_variability = 0.5 # Standard deviation for pace noise
        
        # Points system (2024 standard)
        self.points_map = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
    
    def calculate_base_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate base performance score from pace and grid."""
        df = df.copy()
        
        # Normalize inputs (Rank-based normalization)
        if "best" in df.columns:
            df["pace_score"] = df["best"].rank(ascending=True)
        else:
            # Fallback if no pace data
            df["pace_score"] = df["grid"]
            
        # Combine scores (Lower score = Better performance)
        df["base_score"] = (
            (df["grid"] * self.grid_weight) + 
            (df["pace_score"] * self.pace_weight)
        )
        return df

    def predict(self, features: pd.DataFrame, n_sims: int = 1000) -> pd.DataFrame:
        """
        Run Monte Carlo simulation to predict race results.
        
        Args:
            features: DataFrame with ['driver', 'grid', 'best']
            n_sims: Number of simulation iterations
            
        Returns:
            DataFrame with prediction stats
        """
        # Prepare data
        sim_df = self.calculate_base_score(features)
        drivers = sim_df["driver"].values
        base_scores = sim_df["base_score"].values
        
        # Storage for simulation results
        # Shape: (n_sims, n_drivers)
        sim_ranks = np.zeros((n_sims, len(drivers)))
        
        for i in range(n_sims):
            # 1. Add Performance Noise
            noise = np.random.normal(0, self.pace_variability, size=len(drivers))
            iter_scores = base_scores + noise
            
            # 2. Simulate DNFs
            dnf_mask = np.random.random(len(drivers)) < self.base_dnf_prob
            iter_scores[dnf_mask] = float('inf')
            
            # 3. Determine Positions
            ranks = np.argsort(iter_scores)
            finish_positions = np.empty(len(drivers))
            
            for pos, driver_idx in enumerate(ranks):
                finish_positions[driver_idx] = pos + 1
            
            sim_ranks[i] = finish_positions

        # Analyze Results
        stats = []
        for i, driver in enumerate(drivers):
            driver_results = sim_ranks[:, i]
            
            # Calculate metrics
            win_prob = np.mean(driver_results == 1)
            podium_prob = np.mean(driver_results <= 3)
            q_prob = np.mean(driver_results <= 10)
            avg_pos = np.mean(driver_results)
            
            # Expected Points
            total_points = sum(self.points_map.get(pos, 0) for pos in driver_results)
            exp_points = total_points / n_sims
            
            stats.append({
                "Driver": driver,
                "Grid": features.iloc[i]["grid"],
                "Win %": win_prob,
                "Podium %": podium_prob,
                "Points %": q_prob,
                "Exp. Points": exp_points,
                "Avg Finish": avg_pos
            })
            
        results_df = pd.DataFrame(stats)
        results_df = results_df.sort_values(
            by=["Win %", "Exp. Points"], 
            ascending=[False, False]
        )
        
        return results_df


class F1MLPredictor:
    """
    ML-based F1 Race Predictor using XGBoost with SHAP explainability.
    
    Features used:
    - Grid position (normalized)
    - Practice pace (normalized)
    - Pace consistency (std dev of practice times)
    - Grid vs pace delta (team performance indicator)
    - Position momentum (qualifying progression)
    
    Provides:
    - Position predictions
    - Win/Podium probabilities via calibrated simulation
    - SHAP feature importance and explanations
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            "grid_norm",
            "pace_norm", 
            "pace_consistency",
            "grid_pace_delta",
            "position_strength"
        ]
        self.feature_descriptions = {
            "grid_norm": "Qualifying Position (normalized 0-1, lower is better)",
            "pace_norm": "Practice Pace (normalized 0-1, lower is better)",
            "pace_consistency": "Pace Consistency (lower std = more consistent)",
            "grid_pace_delta": "Grid vs Pace Delta (negative = better pace than grid)",
            "position_strength": "Combined Position Strength Score"
        }
        
        # Points system
        self.points_map = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # SHAP storage
        self.shap_values = None
        self.shap_explainer = None
        
    def _engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Engineer ML features from raw data."""
        features_df = df.copy()
        n_drivers = len(features_df)
        
        features_df["grid_norm"] = (features_df["grid"] - 1) / max(n_drivers - 1, 1)
        
        if "best" in features_df.columns:
            pace_min = features_df["best"].min()
            pace_max = features_df["best"].max()
            pace_range = pace_max - pace_min if pace_max > pace_min else 1
            features_df["pace_norm"] = (features_df["best"] - pace_min) / pace_range
        else:
            features_df["pace_norm"] = features_df["grid_norm"]
        
        # 3. Pace consistency
        if "avg" in features_df.columns and "best" in features_df.columns:
            features_df["pace_consistency"] = (features_df["avg"] - features_df["best"]).clip(0, 5) / 5
        else:
            features_df["pace_consistency"] = 0.2
        
        # 4. Grid vs Pace delta
        raw_delta = features_df["grid_norm"] - features_df["pace_norm"]
        features_df["grid_pace_delta"] = raw_delta * 0.3
        
        # 5. Position strength
        features_df["position_strength"] = (
            0.85 * features_df["grid_norm"] +
            0.10 * features_df["pace_norm"] +
            0.05 * features_df["pace_consistency"]
        )
        
        X = features_df[self.feature_names].values
        return features_df, X
    
    def train_internal_model(self, X: np.ndarray, n_drivers: int):
        """Train internal model using synthetic data."""
        if not HAS_SKLEARN_GB:
            return False
            
        np.random.seed(42)
        n_samples = 2000
        
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            # Generate correlated features
            grid_norm = np.random.uniform(0, 1)
            
            pace_noise = np.random.normal(0, 0.25)
            pace_norm = np.clip(grid_norm + pace_noise, 0, 1)
            
            consistency = np.random.uniform(0, 0.3)
            
            delta = (grid_norm - pace_norm) * 0.3
            
            strength = (
                0.85 * grid_norm +
                0.10 * pace_norm +
                0.05 * consistency
            )
            
            # Simulate finishing position
            base_pos = 1 + grid_norm * (n_drivers - 1)
            
            pace_adjustment = (grid_norm - pace_norm) * 0.5
            
            race_noise = np.random.normal(0, 1.2)
            
            # First lap incidents (10% chance)
            if np.random.random() < 0.10:
                race_noise += np.random.normal(0, 2.5)
            
            # DNF simulation (5% chance)
            if np.random.random() < 0.05:
                final_pos = n_drivers
            else:
                final_pos = base_pos + pace_adjustment + race_noise
                final_pos = np.clip(final_pos, 1, n_drivers)
            
            X_train.append([grid_norm, pace_norm, consistency, delta, strength])
            y_train.append(final_pos)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        
        # Train RandomForest model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        return True
    
    def predict(self, features: pd.DataFrame, n_sims: int = 1000) -> Dict[str, Any]:
        """
        Run ML prediction with SHAP explanations.
        
        Args:
            features: DataFrame with ['driver', 'grid', 'best']
            n_sims: Number of simulation iterations for probability estimation
            
        Returns:
            Dictionary with predictions, SHAP values, and feature importance
        """
        n_drivers = len(features)
        features_df, X = self._engineer_features(features)
        
        # Train internal model for this race
        if HAS_XGBOOST:
            self.train_internal_model(X, n_drivers)
        
        # Get base predictions
        if self.model is not None:
            raw_ml_preds = self.model.predict(X)
            
            # Weighted average of ML and simple physics-based model
            physics_preds = 1 + features_df["position_strength"].values * (n_drivers - 1)
            
            # Combine: 30% ML, 70% Physics
            predicted_positions = 0.3 * raw_ml_preds + 0.7 * physics_preds
            
            predicted_positions = np.clip(predicted_positions, 1, n_drivers)
        else:
            # Fallback: position strength-based prediction
            predicted_positions = 1 + features_df["position_strength"].values * (n_drivers - 1)
        
        # Calculate SHAP values if available
        shap_data = None
        feature_importance = None
        
        if HAS_SHAP and self.model is not None:
            try:
                # Use TreeExplainer for GradientBoosting (sklearn is supported)
                self.shap_explainer = shap.Explainer(self.model, X)
                shap_explanation = self.shap_explainer(X)
                self.shap_values = shap_explanation.values
                
                # Feature importance from SHAP
                feature_importance = pd.DataFrame({
                    "Feature": self.feature_names,
                    "Importance": np.abs(self.shap_values).mean(axis=0),
                    "Description": [self.feature_descriptions[f] for f in self.feature_names]
                }).sort_values("Importance", ascending=False)
                
                # Per-driver SHAP contributions
                shap_data = {
                    "values": self.shap_values,
                    "base_value": shap_explanation.base_values[0] if hasattr(shap_explanation.base_values, '__iter__') else shap_explanation.base_values,
                    "feature_names": self.feature_names,
                    "X": X
                }
            except Exception as e:
                print(f"SHAP calculation failed: {e}")
        
        # Run Monte Carlo simulation for probability estimation
        sim_ranks = np.zeros((n_sims, n_drivers))
        
        for i in range(n_sims):
            # Add prediction noise
            noise = np.random.normal(0, 1.5, size=n_drivers)
            iter_positions = predicted_positions + noise
            
            # DNF simulation
            dnf_mask = np.random.random(n_drivers) < 0.05
            iter_positions[dnf_mask] = float('inf')
            
            # Convert to ranks
            ranks = np.argsort(iter_positions)
            finish_positions = np.empty(n_drivers)
            for pos, driver_idx in enumerate(ranks):
                finish_positions[driver_idx] = pos + 1
            
            sim_ranks[i] = finish_positions
        
        # Build results
        stats = []
        for i, driver in enumerate(features["driver"].values):
            driver_results = sim_ranks[:, i]
            
            win_prob = np.mean(driver_results == 1)
            podium_prob = np.mean(driver_results <= 3)
            points_prob = np.mean(driver_results <= 10)
            avg_pos = np.mean(driver_results)
            
            total_points = sum(self.points_map.get(int(pos), 0) for pos in driver_results)
            exp_points = total_points / n_sims
            
            stats.append({
                "Driver": driver,
                "Grid": int(features.iloc[i]["grid"]),
                "Predicted Pos": round(predicted_positions[i], 1),
                "Win %": win_prob,
                "Podium %": podium_prob,
                "Points %": points_prob,
                "Exp. Points": exp_points,
                "Avg Finish": avg_pos
            })
        
        results_df = pd.DataFrame(stats)
        results_df = results_df.sort_values(
            by=["Win %", "Exp. Points"], 
            ascending=[False, False]
        )
        
        return {
            "predictions": results_df,
            "feature_importance": feature_importance,
            "shap_data": shap_data,
            "features_used": features_df[self.feature_names + ["driver"]],
            "model_type": "Random Forest" if HAS_SKLEARN_GB else "Heuristic"
        }
    
    def get_driver_explanation(self, driver_idx: int, features_df: pd.DataFrame) -> Dict:
        """Get detailed SHAP explanation for a specific driver."""
        if self.shap_values is None:
            return None
        
        driver_shap = self.shap_values[driver_idx]
        driver_features = features_df.iloc[driver_idx]
        
        explanations = []
        for i, feature_name in enumerate(self.feature_names):
            explanations.append({
                "feature": feature_name,
                "description": self.feature_descriptions[feature_name],
                "value": driver_features[feature_name],
                "shap_contribution": driver_shap[i],
                "impact": "positive" if driver_shap[i] > 0 else "negative"
            })
        
        return {
            "driver": driver_features.get("driver", f"Driver {driver_idx}"),
            "base_prediction": self.shap_explainer.expected_value if self.shap_explainer else None,
            "feature_contributions": sorted(explanations, key=lambda x: abs(x["shap_contribution"]), reverse=True)
        }


def get_shap_summary_data(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Convert SHAP values to a format suitable for visualization."""
    if shap_values is None:
        return None
    
    # Create summary dataframe
    summary = []
    for i, feature in enumerate(feature_names):
        summary.append({
            "Feature": feature,
            "Mean |SHAP|": np.abs(shap_values[:, i]).mean(),
            "Max |SHAP|": np.abs(shap_values[:, i]).max(),
            "Std SHAP": shap_values[:, i].std()
        })
    
    return pd.DataFrame(summary).sort_values("Mean |SHAP|", ascending=False)