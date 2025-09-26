import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_session_data, aggregate_sector_times, prepare_qualifying_data
from src.model import F1RacePredictor
from src.config import WEATHER_API_KEY, RACE_SCHEDULE

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Race Predictor")
st.markdown("Predict F1 race results using machine learning")

# Sidebar for configuration
with st.sidebar:
    st.header("Race Selection")
    
    # Race selection
    race_options = [race["name"] for race in RACE_SCHEDULE]
    selected_race_name = st.selectbox("Select Race", options=race_options, index=7)  # Default to Monaco
    selected_race = next((race for race in RACE_SCHEDULE if race["name"] == selected_race_name), RACE_SCHEDULE[7])  # Default to Monaco
    
    st.subheader(f"Selected: {selected_race['name']} Grand Prix")
    st.markdown(f"**Location:** {selected_race['location']}")
    st.markdown(f"**Date:** {selected_race['date']}")
    st.markdown(f"**Round:** {selected_race['round']}")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Weather API key input
    api_key = st.text_input(
        "OpenWeather API Key", 
        value=WEATHER_API_KEY if WEATHER_API_KEY != "YOURAPIKEY" else "",
        type="password",
        help="Enter your OpenWeatherMap API key for weather data"
    )
    
    # Model parameters
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of Estimators", 50, 200, 100)
    learning_rate = st.slider("Learning Rate", 0.1, 1.0, 0.7)
    max_depth = st.slider("Max Depth", 2, 10, 3)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Data Loading")
    
    with st.spinner(f"Loading {selected_race['name']} session data..."):
        try:
            laps_2024 = load_session_data(race_name=selected_race['name'])
            sector_times_2024 = aggregate_sector_times(laps_2024)
            st.success(f"✅ Loaded data for {len(laps_2024['Driver'].unique())} drivers")
        except Exception as e:
            st.error(f"Error loading session data: {e}")
            st.stop()

with col2:
    st.header("🌤️ Weather & Qualifying Data")
    
    with st.spinner(f"Preparing {selected_race['name']} qualifying data..."):
        qualifying_2025, rain_probability, temperature = prepare_qualifying_data(selected_race['name'])
        
        st.metric("Rain Probability", f"{rain_probability*100:.0f}%")
        st.metric("Temperature", f"{temperature:.1f}°C")

# Model training and prediction
st.header("🤖 Model Training & Predictions")

if st.button("Run Prediction", type="primary"):
    with st.spinner("Training model and making predictions..."):
        # Initialize predictor
        predictor = F1RacePredictor()
        predictor.model.n_estimators = n_estimators
        predictor.model.learning_rate = learning_rate
        predictor.model.max_depth = max_depth
        
        # Prepare data
        merged_data, X, y = predictor.prepare_data(
            qualifying_2025, sector_times_2024, laps_2024, 
            rain_probability, temperature
        )
        
        # Train and predict
        predictions, mae = predictor.train_and_predict(X, y)
        merged_data["PredictedRaceTime (s)"] = predictions
        
        # Sort results
        final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
        
        # Display results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("🏁 Predicted Race Results")
            
            # Create podium display
            podium = final_results.head(3)
            st.markdown("### 🏆 Podium Finishers")
            
            medals = ["🥇", "🥈", "🥉"]
            for i, (_, driver_data) in enumerate(podium.iterrows()):
                st.markdown(f"{medals[i]} **P{i+1}: {driver_data['Driver']}** - {driver_data['PredictedRaceTime (s)']:.2f}s")
            
            # Full results table
            st.markdown("### Full Results")
            results_df = final_results[["Driver", "Team", "PredictedRaceTime (s)"]].copy()
            results_df.index = range(1, len(results_df) + 1)
            results_df.index.name = "Position"
            st.dataframe(results_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Performance")
            st.metric("Model Error (MAE)", f"{mae:.2f} seconds")
            
            # Feature importance
            st.markdown("### Feature Importance")
            feature_importance = predictor.get_feature_importance()
            fi_df = pd.DataFrame(
                list(feature_importance.items()), 
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=False)
            st.dataframe(fi_df, use_container_width=True)
        
        with col3:
            st.subheader("🎯 Key Insights")
            
            # Fastest predicted time
            fastest = final_results.iloc[0]
            st.info(f"**Fastest Predicted Time:** {fastest['PredictedRaceTime (s)']:.2f}s")
            
            # Time gap to leader
            time_gaps = final_results["PredictedRaceTime (s)"] - fastest["PredictedRaceTime (s)"]
            avg_gap = time_gaps[1:].mean()
            st.info(f"**Average Gap to Leader:** {avg_gap:.2f}s")
            
            # Closest battle
            gaps_between = final_results["PredictedRaceTime (s)"].diff()[1:]
            min_gap_idx = gaps_between.idxmin()
            st.info(f"**Closest Battle:** P{min_gap_idx} vs P{min_gap_idx+1} ({gaps_between[min_gap_idx]:.3f}s)")

# Visualization section
st.header("📊 Visualizations")

if 'final_results' in locals():
    col1, col2 = st.columns(2)
    
    with col1:
        # Clean air race pace effect
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = ax1.scatter(
            final_results["CleanAirRacePace (s)"], 
            final_results["PredictedRaceTime (s)"],
            c=range(len(final_results)),
            cmap='viridis',
            s=100
        )
        
        for i, driver in enumerate(final_results["Driver"]):
            ax1.annotate(
                driver, 
                (final_results["CleanAirRacePace (s)"].iloc[i], 
                 final_results["PredictedRaceTime (s)"].iloc[i]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8
            )
        
        ax1.set_xlabel("Clean Air Race Pace (s)")
        ax1.set_ylabel("Predicted Race Time (s)")
        ax1.set_title("Effect of Clean Air Race Pace on Predicted Results")
        plt.colorbar(scatter, ax=ax1, label="Finishing Position")
        st.pyplot(fig1)
    
    with col2:
        # Feature importance bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        feature_importance = predictor.get_feature_importance()
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        bars = ax2.barh(features, importances, color='skyblue')
        ax2.set_xlabel("Importance")
        ax2.set_title("Feature Importance in Race Time Prediction")
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', 
                    ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Time gaps visualization
    st.subheader("⏱️ Time Gaps Visualization")
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    positions = range(1, len(final_results) + 1)
    time_to_leader = final_results["PredictedRaceTime (s)"] - final_results["PredictedRaceTime (s)"].iloc[0]
    
    bars = ax3.barh(positions, time_to_leader, color='lightcoral')
    ax3.set_yticks(positions)
    ax3.set_yticklabels([f"P{i}: {driver}" for i, driver in zip(positions, final_results["Driver"])])
    ax3.set_xlabel("Time Behind Leader (seconds)")
    ax3.set_title("Time Gaps to Race Leader")
    ax3.invert_yaxis()
    
    # Add time labels
    for bar, time_gap in zip(bars, time_to_leader):
        if time_gap > 0:
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'+{time_gap:.2f}s', 
                    ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig3)

# Footer
st.markdown("---")
st.markdown(f"🏎️ **F1 {selected_race['name']} GP Predictor** | Data source: FastF1 | Weather: OpenWeatherMap")