import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_session_data, aggregate_sector_times, prepare_qualifying_data
from src.model import F1RacePredictor
from src.config import WEATHER_API_KEY, RACE_SCHEDULE, CLEAN_AIR_RACE_PACE, QUALIFYING_2025_DATA

# --- Page Configuration ---
st.set_page_config(
    page_title="F1 Strategy AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .f1-header {
        color: #FF1801;
        font-weight: 800;
        font-family: 'Segoe UI', sans-serif;
    }
    .stat-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF1801;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([1, 4])
with col1:
    st.title("🏎️")
with col2:
    st.markdown("<h1 class='f1-header'>F1 STRATEGY AI <span style='color:white; font-size:0.5em'>PRO</span></h1>", unsafe_allow_html=True)
    st.caption("Advanced Predictive Analytics & Monte Carlo Simulation Engine")

# --- Tabs Structure ---
tab_control, tab_telemetry, tab_simulation = st.tabs(["🏁 Race Control", "📊 Telemetry Dashboard", "🔮 Monte Carlo Analysis"])

# --- Session State Management ---
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None

# ==============================================================================
# TAB 1: RACE CONTROL
# ==============================================================================
with tab_control:
    col_setup, col_strategy = st.columns([1, 1])
    
    with col_setup:
        st.subheader("📍 Grand Prix Setup")
        
        race_options = [race["name"] for race in RACE_SCHEDULE]
        selected_race_name = st.selectbox("Select Circuit", options=race_options, index=7) # Monaco default
        selected_race = next((race for race in RACE_SCHEDULE if race["name"] == selected_race_name), RACE_SCHEDULE[7])
        
        st.info(f"**{selected_race['name']}** | Round {selected_race['round']} | {selected_race['date']}")
        
        st.subheader("🌤️ Environmental Conditions")
        rain_override = st.slider("Rain Probability (%)", 0, 100, 0, 5) / 100.0
        temp_override = st.slider("Track Temperature (°C)", 10, 60, 25, 1)
        weather_override = (rain_override, float(temp_override))

        api_key = st.text_input("Weather API Key", value=WEATHER_API_KEY if WEATHER_API_KEY != "YOURAPIKEY" else "", type="password")

    with col_strategy:
        st.subheader("🏎️ Driver & Strategy Config")
        
        with st.expander("⏱️ Pace & Quali Adjustments", expanded=False):
            st.caption("Fine-tune driver performance relative to baseline.")
            
            # Deep copy data to avoid mutating original
            qualifying_override = {k: v[:] for k, v in QUALIFYING_2025_DATA.items()}
            race_pace_override = CLEAN_AIR_RACE_PACE.copy()
            
            drivers_list = QUALIFYING_2025_DATA["Driver"]
            tabs_drivers = st.tabs(["Top Teams", "Midfield"])
            
            top_teams = ["VER", "NOR", "LEC", "HAM", "SAI", "PIA", "RUS", "ALO"]
            
            with tabs_drivers[0]:
                for driver in top_teams:
                    if driver in drivers_list:
                        c1, c2 = st.columns(2)
                        with c1:
                            pace_delta = st.number_input(f"{driver} Pace (s)", -2.0, 2.0, 0.0, 0.05, key=f"p_{driver}")
                            race_pace_override[driver] += pace_delta
                        with c2:
                            idx = drivers_list.index(driver)
                            curr_q = QUALIFYING_2025_DATA["QualifyingTime (s)"][idx]
                            if curr_q:
                                q_delta = st.number_input(f"{driver} Quali (s)", -2.0, 2.0, 0.0, 0.05, key=f"q_{driver}")
                                qualifying_override["QualifyingTime (s)"][idx] = curr_q + q_delta

        with st.expander("♟️ Tyre Strategy", expanded=True):
            st.caption("Starting compound affects initial pace vs degradation.")
            tyre_strategy = {}
            cols_tyre = st.columns(4)
            key_drivers = ["VER", "NOR", "LEC", "HAM", "SAI", "PIA", "RUS", "ALO"]
            
            for i, driver in enumerate(key_drivers):
                with cols_tyre[i % 4]:
                    tyre = st.selectbox(f"{driver}", ["Soft", "Medium", "Hard"], index=1, key=f"t_{driver}", label_visibility="collapsed")
                    st.caption(driver)
                    tyre_strategy[driver] = tyre

    st.markdown("---")
    if st.button("🚀 INITIALIZE SIMULATION", type="primary", use_container_width=True):
        with st.spinner("Loading telemetry, configuring physics engine..."):
            try:
                laps_2024, results_2024 = load_session_data(race_name=selected_race['name'])
                sector_times_2024 = aggregate_sector_times(laps_2024)
                
                qualifying_2025, rain_prob, temp = prepare_qualifying_data(
                    selected_race['name'], 
                    qualifying_override=qualifying_override,
                    race_pace_override=race_pace_override,
                    weather_override=weather_override,
                    target_date=selected_race['date'],
                    tyre_strategy=tyre_strategy
                )
                
                predictor = F1RacePredictor()
                
                merged_data, X, y = predictor.prepare_data(
                    qualifying_2025, sector_times_2024, laps_2024, results_2024,
                    rain_prob, temp
                )
                
                # Train Model (Baseline)
                base_preds, mae = predictor.train_and_predict(X, y)
                merged_data["PredictedPosition"] = base_preds
                
                st.session_state.predictor = predictor
                st.session_state.simulation_data = (merged_data, X, y)
                st.session_state.mae = mae
                st.session_state.rain_prob = rain_prob
                
                st.success(f"✅ Simulation Initialized! Baseline MAE: {mae:.2f} positions")
                
            except Exception as e:
                st.error(f"Initialization Failed: {e}")
                st.exception(e)

# ==============================================================================
# TAB 2: TELEMETRY DASHBOARD
# ==============================================================================
with tab_telemetry:
    if st.session_state.simulation_data is None:
        st.warning("⚠️ Please Initialize Simulation in 'Race Control' first.")
    else:
        merged_data, X, y = st.session_state.simulation_data
        predictor = st.session_state.predictor
        
        st.header("📊 Telemetry Analysis")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Predicted Finishing Order (Baseline)")
            final_results = merged_data.sort_values("PredictedPosition")
            
            fig_order = px.bar(
                final_results,
                x="Driver",
                y="PredictedPosition",
                color="PredictedPosition",
                color_continuous_scale="RdYlGn_r",
                text_auto='.1f',
                title="Predicted Position Score (Lower is Better)"
            )
            fig_order.update_layout(xaxis_title="Driver", yaxis_title="Position Score")
            st.plotly_chart(fig_order, use_container_width=True)
            
        with c2:
            st.subheader("Model Insights")
            st.metric("Rain Probability", f"{st.session_state.rain_prob:.0%}")
            st.metric("Model Uncertainty", f"±{st.session_state.mae:.1f} pos")
            
            imp = predictor.get_feature_importance()
            df_imp = pd.DataFrame(list(imp.items()), columns=["Feature", "Value"]).sort_values("Value")
            fig_imp = px.bar(df_imp, x="Value", y="Feature", orientation='h', title="Factor Weighting")
            fig_imp.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# TAB 3: MONTE CARLO ANALYSIS
# ==============================================================================
with tab_simulation:
    if st.session_state.simulation_data is None:
        st.warning("⚠️ Please Initialize Simulation in 'Race Control' first.")
    else:
        merged_data, X, y = st.session_state.simulation_data
        predictor = st.session_state.predictor
        drivers = merged_data["Driver"]
        
        st.header("🔮 Monte Carlo Simulation")
        st.markdown("""
        **Methodology:** This engine runs **1,000** race simulations. In each iteration, it introduces Gaussian noise to:
        1.  **Race Pace** (Simulating driver consistency, traffic, errors)
        2.  **Pit Stops** (Simulating crew performance and strategy calls)
        """)
        
        if st.button("🎰 RUN 1,000 SIMULATIONS", type="primary"):
            with st.spinner("Simulating 1,000 parallel universes..."):
                mc_results = predictor.monte_carlo_predict(X, drivers, n_simulations=1000)
                st.session_state.mc_results = mc_results
                st.success("Analysis Complete!")
        
        if st.session_state.mc_results is not None:
            mc_df = st.session_state.mc_results
            
            winner = mc_df.iloc[0]
            
            col_win, col_podium, col_cons = st.columns(3)
            with col_win:
                st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                st.metric("Most Likely Winner", winner['Driver'], f"{winner['Win Probability']:.1%} Prob")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_podium:
                st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
                podium_contenders = mc_df[mc_df['Podium Probability'] > 0.3]['Driver'].tolist()
                st.write("**High Podium Probability:**")
                st.write(", ".join(podium_contenders))
                st.markdown("</div>", unsafe_allow_html=True)

            # Win Probability Chart
            st.subheader("🏆 Win Probability Distribution")
            
            # Filter reasonable contenders
            contenders = mc_df[mc_df['Win Probability'] > 0.01]
            
            fig_win = px.pie(
                contenders, 
                values='Win Probability', 
                names='Driver', 
                title='Championship Probability',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_win, use_container_width=True)
            
            # Confidence Intervals (Candlestick/Box-like)
            st.subheader("📊 Performance Confidence Intervals (P5 - Avg - P95)")
            
            fig_conf = go.Figure()
            
            mc_df_sorted = mc_df.sort_values("Avg Finish")
            
            fig_conf.add_trace(go.Scatter(
                x=mc_df_sorted['Driver'], 
                y=mc_df_sorted['Avg Finish'],
                mode='markers',
                name='Average Finish',
                marker=dict(color='white', size=8)
            ))
            
            for i, row in mc_df_sorted.iterrows():
                fig_conf.add_trace(go.Scatter(
                    x=[row['Driver'], row['Driver']],
                    y=[row['Best Case (P5)'], row['Worst Case (P95)']],
                    mode='lines',
                    line=dict(color='#FF1801', width=3),
                    showlegend=False
                ))
            
            fig_conf.update_layout(
                title="Driver Finishing Range (5th to 95th Percentile)",
                yaxis_title="Position (Lower is Better)",
                yaxis=dict(autorange="reversed"),
                template="plotly_dark"
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            
            st.subheader("📄 Detailed Simulation Data")
            st.dataframe(mc_df.style.format({
                "Win Probability": "{:.1%}",
                "Podium Probability": "{:.1%}",
                "Avg Finish": "{:.2f}",
                "Best Case (P5)": "{:.0f}",
                "Worst Case (P95)": "{:.0f}"
            }), use_container_width=True)


# Footer
st.markdown("---")
st.markdown(f"🏎️ **F1 {selected_race['name']} GP Predictor** | Data source: FastF1 | Weather: OpenWeatherMap")