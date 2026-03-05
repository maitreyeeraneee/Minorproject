import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def weight_tracker_tab():
    st.header("⚖️ Weight Tracking")

    if not st.session_state.metrics_calculated:
        st.info("Please calculate your needs in the sidebar to see weight tracking.")
        return

    # Current metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Weight", f"{st.session_state.get('current_weight', 'N/A')} kg")
    with col2:
        st.metric("Ideal Body Weight", f"{st.session_state.ideal_weight:.1f} kg")
    with col3:
        st.metric("Target Weight", f"{st.session_state.get('target_weight', st.session_state.ideal_weight):.1f} kg")

    # Initialize weight log if not exists
    if 'weight_log' not in st.session_state:
        st.session_state.weight_log = {}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Log Weight")

        # Weight input
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=st.session_state.get('current_weight', 70.0))
        date = st.date_input("Date", datetime.now().date())

        if st.button("Log Weight"):
            log_weight(weight, date)

        # Set target weight
        st.subheader("Set Target Weight")
        target_weight = st.number_input("Target Weight (kg)", min_value=30.0, max_value=200.0, value=st.session_state.get('target_weight', st.session_state.ideal_weight))
        if st.button("Set Target"):
            st.session_state.target_weight = target_weight
            st.success(f"Target weight set to {target_weight} kg")

    with col2:
        # Weight history chart
        if st.session_state.weight_log:
            df = pd.DataFrame(list(st.session_state.weight_log.items()), columns=['Date', 'Weight'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            # Update current weight
            st.session_state.current_weight = df['Weight'].iloc[-1]

            # Line chart with target line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Weight'], mode='lines+markers', name='Weight', line=dict(color='#4CAF50')))
            fig.add_hline(y=st.session_state.get('target_weight', st.session_state.ideal_weight), line_dash="dash", line_color="red", annotation_text="Target Weight")
            fig.add_hline(y=st.session_state.ideal_weight, line_dash="dot", line_color="blue", annotation_text="Ideal Weight")

            fig.update_layout(title="Weight Change Over Time", xaxis_title="Date", yaxis_title="Weight (kg)")
            st.plotly_chart(fig, use_container_width=True)

            # Weight change stats
            if len(df) > 1:
                initial_weight = df['Weight'].iloc[0]
                current_weight = df['Weight'].iloc[-1]
                change = current_weight - initial_weight
                st.metric("Total Change", f"{change:+.1f} kg", f"From {initial_weight:.1f} kg")

                # Weekly trend
                df['week'] = df['Date'].dt.to_period('W')
                weekly_avg = df.groupby('week')['Weight'].mean().reset_index()
                weekly_avg['week'] = weekly_avg['week'].astype(str)
                if len(weekly_avg) > 1:
                    weekly_change = weekly_avg['Weight'].iloc[-1] - weekly_avg['Weight'].iloc[-2]
                    st.metric("Weekly Trend", f"{weekly_change:+.2f} kg")

                # Monthly trend
                df['month'] = df['Date'].dt.to_period('M')
                monthly_avg = df.groupby('month')['Weight'].mean().reset_index()
                monthly_avg['month'] = monthly_avg['month'].astype(str)
                if len(monthly_avg) > 1:
                    monthly_change = monthly_avg['Weight'].iloc[-1] - monthly_avg['Weight'].iloc[-2]
                    st.metric("Monthly Trend", f"{monthly_change:+.2f} kg")
        else:
            st.info("No weight data logged yet. Start logging your weight to see charts!")

def log_weight(weight, date):
    date_str = date.strftime('%Y-%m-%d')
    st.session_state.weight_log[date_str] = weight
    st.session_state.current_weight = weight
    st.success(f"Weight logged: {weight} kg for {date.strftime('%B %d, %Y')}")
    st.rerun()
