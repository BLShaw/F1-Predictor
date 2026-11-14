# F1 Predictor

A Formula 1 prediction engine combining Monte Carlo simulation with Machine Learning for Grand Prix outcomes.

[![CI](https://github.com/BLShaw/f1-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/BLShaw/f1-predictor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-51%20passed-10B981.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Architecture

```mermaid
flowchart LR
    subgraph Data["1. Telemetry & Regulations"]
        A["FastF1 API Ingestion"] --> B["Clean Flying Laps\n(pick_quicklaps + IsAccurate)"]
        B --> C["10-Feature Matrix\n(Grid, Quali Gap, Practice, Form, Track)"]
    end

    subgraph Model["2. Machine Learning & Simulation"]
        C --> D["Random Forest Regressor\n(300 Trees, Max Depth 6, Classified Only)"]
        D --> E["Vectorized Monte Carlo Engine\n(Circuit, Wet-Weather & Team DNF Calibration)"]
        E --> F["Points Withholding Rule\n(DNFs Excluded from Top-10 Points)"]
    end

    subgraph Serving["3. Dashboard & Explainability"]
        F --> G["Streamlit Mission Control\n(Overview, Backtest & Dynamic Model Spec)"]
        F --> H["TreeSHAP Explainability\n(Global Impact & Driver Waterfall)"]
    end
```

---

## Key Features

- **Grand Prix Centric**: Organizes session data by Season and Round across Free Practice, Qualifying, Sprint Shootout, Sprint Race, and Grand Prix, plus circuit technical metadata (`length_km`, `corners`, `downforce`, `overtaking_difficulty`, `tire_wear`, `type`).
- **10-Feature ML & Vectorized Monte Carlo Engine**: Random Forest Regressor trained across 10 pre-race features (`grid_norm`, `quali_gap_norm`, `pace_norm`, `pace_consistency`, lagged pre-race EWMA `driver_form` and `constructor_form`, `overtaking_difficulty`, `is_street_circuit`, `has_practice_data`, `is_sprint`) paired with a seeded, vectorized NumPy simulator calibrating variance by circuit overtaking difficulty and `weather.rainfall`, and DNF rates by constructor reliability.
- **Non-Finisher Points Withholding**: Simulated DNFs are strictly excluded from top-10 points allocation, advancing classified finishers into vacated championship points positions.
- **SHAP Explainability & Backtesting**: Global feature sensitivity charts, per-driver Plotly SHAP waterfall charts (`Base Value` -> feature pushes -> `Predicted Pos`), and an automatic **Predicted vs. Actual** backtest view when official race results exist.
- **Regulatory Rules Support**: Full support for Sprint Weekends, dynamically handling 2022 sprint rules (Saturday sprint sets Sunday grid) versus 2023+ regulations (qualifying sets Sunday grid; shootout sets sprint grid), and 20 vs. 22 car knockout thresholds.
- **Decoupled Architecture & Live Metadata Sync**: Framework-agnostic domain and modeling logic separated from Streamlit presentation/caching adapters, with `scripts/train_historical_model.py` exporting `data/models/model_metadata.json` so dashboard model specification cards stay synchronized with the latest training run.

---

## Model Performance

The core ML engine (`RandomForestRegressor`, `max_depth=6`, `min_samples_leaf=4`, `max_features=0.8`) is trained on historical telemetry, qualifying gaps, rolling form, and circuit metadata from 2022 to the present across temporal out-of-time splits (`Train`: 2022–2024 [1,165 samples], `Validation`: 2025 [419 samples], `Test`: 2026 [348 samples]):

| Benchmark Metric | 2025 Validation (419 Samples) | 2026 Out-of-Time Test (348 Samples) | 2026 Baseline Comparison |
| :--- | :--- | :--- | :--- |
| **Points Finish Accuracy (Top 10)** | **79.7%** (F1: 0.815) | **81.3%** (F1: 0.836) | — |
| **Podium Finish Accuracy (Top 3)** | **91.4%** (F1: 0.705) | **88.5%** (F1: 0.643) | 82.8% Dummy Prior (**+5.7%**) |
| **Mean Absolute Error (MAE)** | **2.30 positions** | **2.21 positions** | 2.34 positions Grid-Only (**-0.13 pos**) |
| **Root Mean Squared Error (RMSE)** | **3.01 positions** | **2.84 positions** | — |
| **Explained Variance ($R^2$)** | **0.662** | **0.703** | — |

*Target labels during training are strictly restricted to classified race finishers to prevent mechanical retirement label corruption. Running `scripts/train_historical_model.py` exports these hyperparameters, partition counts, and evaluation metrics to `data/models/model_metadata.json`.*

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/BLShaw/f1-predictor.git
cd f1-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or editable install with development tools (pytest, ruff, bandit):
pip install -e ".[dev]"
```

### 3. Run Quality Verification & Tests
```bash
ruff check .
pytest -v
```

### 4. Run Application
```bash
streamlit run app.py
```

---

## CLI Pipelines

```bash
# Ingest historical session telemetry (2022 to present)
python scripts/download_historical_data.py

# Retrain model and evaluate temporal out-of-time splits
python scripts/train_historical_model.py
```

---

## Screenshots

<img width="1920" height="1080" alt="Screenshot_40" src="https://github.com/user-attachments/assets/ccdb8847-c4ec-463e-825a-b4ee7b6f65d5" />
<img width="1920" height="1080" alt="Screenshot_41" src="https://github.com/user-attachments/assets/86803239-47d7-4dec-ba5c-29889e871751" />
<img width="1920" height="1080" alt="Screenshot_42" src="https://github.com/user-attachments/assets/331de847-e99c-426f-98b6-1747ec417350" />
<img width="1920" height="1080" alt="Screenshot_43" src="https://github.com/user-attachments/assets/20e88f05-defd-46b1-bf75-8c2678a8d866" />
<img width="1920" height="1080" alt="Screenshot_44" src="https://github.com/user-attachments/assets/10f20cd4-d8da-4ee6-866c-94d073e98d7f" />
<img width="1920" height="1080" alt="Screenshot_45" src="https://github.com/user-attachments/assets/8ccc3184-ea70-4278-92e1-439476f0bc12" />
<img width="1920" height="1080" alt="Screenshot_46" src="https://github.com/user-attachments/assets/b44353b0-508e-46be-9221-685ce894665e" />
<img width="1920" height="1080" alt="Screenshot_47" src="https://github.com/user-attachments/assets/68467761-4208-4dca-8b22-6a92dcc4e1a5" />
<img width="1920" height="1080" alt="Screenshot_48" src="https://github.com/user-attachments/assets/d540634d-fffb-46ff-a5c0-039a610a9b89" />

---

## License

MIT License. See [LICENSE](LICENSE) for details.
