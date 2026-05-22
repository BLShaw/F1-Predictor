import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetcher import DATA_DIR
from src.data_loader import aggregate_practice_pace
from src.model import F1MLPredictor

def load_json(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def extract_race_results(race_data):
    if not race_data or 'results' not in race_data:
        return {}
    results = {}
    for row in race_data['results']:
        driver = row.get('driver')
        pos = row.get('position')
        if driver and pos:
            try:
                results[driver] = int(float(pos))
            except ValueError:
                # DNF, NC, etc. - assign a high position
                results[driver] = 20
    return results

def extract_grid_positions(quali_data):
    if not quali_data or 'results' not in quali_data:
        return {}
    results = {}
    for row in quali_data['results']:
        driver = row.get('driver')
        pos = row.get('position')
        if driver and pos:
            try:
                results[driver] = int(float(pos))
            except ValueError:
                results[driver] = 20
    return results

def main():
    print("Starting Historical Model Training Pipeline")
    
    # 1. Load all historical GPs
    seasons_dir = Path(DATA_DIR)
    
    if not seasons_dir.exists():
        print(f"Data directory not found: {seasons_dir}")
        return

    all_features = []
    all_targets = []
    
    total_gps = 0
    
    print("Extracting features from historical races...")
    
    # Initialize predictor once outside the loop to prevent spam
    predictor = F1MLPredictor()
    
    # Iterate through all seasons and GPs
    for year_dir in sorted(seasons_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
            
        for gp_dir in sorted(year_dir.iterdir()):
            if not gp_dir.is_dir():
                continue
                
            # Load necessary files
            fp1 = load_json(gp_dir / "fp1.json")
            fp2 = load_json(gp_dir / "fp2.json")
            fp3 = load_json(gp_dir / "fp3.json")
            quali = load_json(gp_dir / "qualifying.json")
            sprint_quali = load_json(gp_dir / "sprint_qualifying.json")
            sprint_shootout = load_json(gp_dir / "sprint_shootout.json")
            race = load_json(gp_dir / "race.json")
            
            if not race:
                continue # Can't train without targets
                
            # Prefer standard qualifying, fallback to sprint quali/shootout for grid
            grid_data = quali if quali else (sprint_quali if sprint_quali else sprint_shootout)
            if not grid_data:
                continue
                
            grid_positions = extract_grid_positions(grid_data)
            race_positions = extract_race_results(race)
            
            # Aggregate practice pace
            gp_data = {"sessions": {}}
            if fp1: gp_data["sessions"]["fp1"] = fp1
            if fp2: gp_data["sessions"]["fp2"] = fp2
            if fp3: gp_data["sessions"]["fp3"] = fp3
            
            pace_df = aggregate_practice_pace(gp_data)
            
            if pace_df.empty:
                continue
                
            # Combine into a single DataFrame for this GP
            drivers = list(race_positions.keys())
            
            gp_features = []
            for d in drivers:
                if d not in grid_positions:
                    continue
                    
                row = {
                    "driver": d,
                    "grid": grid_positions[d],
                }
                
                # Add pace data
                pace_row = pace_df[pace_df["driver"] == d]
                if not pace_row.empty:
                    row["best"] = pace_row.iloc[0].get("best", None)
                    row["avg"] = pace_row.iloc[0].get("avg", None)
                else:
                    # Missing practice data, assume average or back of pack
                    continue
                    
                if pd.isna(row["best"]) or pd.isna(row["avg"]):
                    continue
                    
                gp_features.append(row)
                all_targets.append(race_positions[d])
            
            if not gp_features:
                continue
                
            gp_df = pd.DataFrame(gp_features)
            
            # Use F1MLPredictor to engineer the features
            # _engineer_features expects columns: driver, grid, best, avg
            features_df, X_gp = predictor._engineer_features(gp_df)
            
            # Add the engineered rows to our training set
            for x_row in X_gp:
                all_features.append(x_row)
                
            total_gps += 1
            
    if not all_features:
        print("No valid training data found.")
        return
        
    X_train = np.array(all_features)
    y_train = np.array(all_targets)
    
    print(f"Extracted data from {total_gps} Grand Prix events.")
    print(f"Total training samples: {len(X_train)} driver performances.")
    
    # 2. Train the Model
    print("Training RandomForest model...")
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    score = model.score(X_train, y_train)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
    
    y_pred = model.predict(X_train)
    
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    # Round predictions to nearest integer for classification metrics
    y_pred_rounded = np.clip(np.round(y_pred), 1, 20)
    
    exact_accuracy = accuracy_score(y_train, y_pred_rounded)
    
    # Podium predictions (Top 3)
    y_train_podium = (y_train <= 3)
    y_pred_podium = (y_pred_rounded <= 3)
    podium_accuracy = accuracy_score(y_train_podium, y_pred_podium)
    podium_f1 = f1_score(y_train_podium, y_pred_podium)
    
    # Points finish (Top 10)
    y_train_points = (y_train <= 10)
    y_pred_points = (y_pred_rounded <= 10)
    points_accuracy = accuracy_score(y_train_points, y_pred_points)
    points_f1 = f1_score(y_train_points, y_pred_points)
    
    print("\n--- Detailed Model Performance Metrics ---")
    print(f"R² Score:              {score:.3f}")
    print(f"Mean Absolute Error:   {mae:.2f} positions")
    print(f"Root Mean Squared Err: {rmse:.2f} positions")
    print(f"Exact Position Acc:    {exact_accuracy*100:.1f}%")
    print(f"Podium Prediction Acc: {podium_accuracy*100:.1f}% (F1: {podium_f1:.3f})")
    print(f"Points Finish Acc:     {points_accuracy*100:.1f}% (F1: {points_f1:.3f})")
    print("------------------------------------------\n")
    
    # 3. Save the Model
    models_dir = Path(DATA_DIR).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "f1_historical_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"Model successfully saved to {model_path}")
    print("You can now restart the Streamlit app to use the real-data model!")

if __name__ == "__main__":
    main()
