import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def water_tracker_tab():
    st.header("💧 Water Tracking")

    if not st.session_state.metrics_calculated:
        st.info("Please calculate your needs in the sidebar to see water tracking.")
        return

    # Daily target in glasses (assuming 250ml per glass)
    glasses_target = int(st.session_state.water_intake / 250)
    st.metric("Daily Water Target", f"{glasses_target} glasses ({st.session_state.water_intake:.0f} ml)")

    # Initialize water log if not exists
    if 'water_log' not in st.session_state:
        st.session_state.water_log = {}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Log Water Intake")

        # Quick add buttons
        glass_sizes = [1, 2, 3]  # glasses
        cols = st.columns(3)
        for i, size in enumerate(glass_sizes):
            with cols[i]:
                if st.button(f"Add {size} glass{'es' if size > 1 else ''}", key=f"water_{size}"):
                    add_water_intake(size * 250)  # 250ml per glass

        # Custom amount
        custom_amount = st.number_input("Custom Amount (ml)", min_value=50, max_value=1000, value=250, step=50)
        if st.button("Add Custom"):
            add_water_intake(custom_amount)

    with col2:
        # Today's progress
        today = datetime.now().date()
        today_str = today.strftime('%Y-%m-%d')

        if today_str in st.session_state.water_log:
            total_ml = sum(entry['amount'] for entry in st.session_state.water_log[today_str])
            glasses_consumed = int(total_ml / 250)
            progress = min(total_ml / st.session_state.water_intake, 1.0)

            # Progress ring
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=glasses_consumed,
                title={'text': f"Glasses Today ({glasses_consumed}/{glasses_target})"},
                gauge={'axis': {'range': [0, glasses_target]},
                       'bar': {'color': "#2196F3"},
                       'steps': [{'range': [0, glasses_target], 'color': "#E3F2FD"}]}))
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Today's Intake", f"{total_ml:.0f} ml", f"{progress*100:.0f}% of target")
            st.progress(progress)

            # Today's log
            st.subheader("Today's Log")
            for entry in st.session_state.water_log[today_str]:
                st.write(f"🕒 {entry['time']}: {entry['amount']} ml")
        else:
            st.info("No water logged today yet.")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                title={'text': f"Glasses Today (0/{glasses_target})"},
                gauge={'axis': {'range': [0, glasses_target]},
                       'steps': [{'range': [0, glasses_target], 'color': "#E3F2FD"}]}))
            st.plotly_chart(fig, use_container_width=True)

    # Analytics
    if st.session_state.water_log:
        st.header("📊 Hydration Analytics")

        # Prepare data
        data = []
        for date, entries in st.session_state.water_log.items():
            total = sum(entry['amount'] for entry in entries)
            data.append({'date': date, 'total_ml': total, 'glasses': int(total / 250)})

        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Daily bar chart
            fig_daily = px.bar(df, x='date', y='glasses', title="Daily Water Intake (Glasses)")
            st.plotly_chart(fig_daily, use_container_width=True)

            # Weekly summary
            df['week'] = df['date'].dt.to_period('W').astype(str)
            weekly = df.groupby('week')['glasses'].sum().reset_index()
            fig_weekly = px.bar(weekly, x='week', y='glasses', title="Weekly Water Intake (Glasses)")
            st.plotly_chart(fig_weekly, use_container_width=True)

            # Monthly summary
            df['month'] = df['date'].dt.to_period('M').astype(str)
            monthly = df.groupby('month')['glasses'].sum().reset_index()
            fig_monthly = px.bar(monthly, x='month', y='glasses', title="Monthly Water Intake (Glasses)")
            st.plotly_chart(fig_monthly, use_container_width=True)

def add_water_intake(amount):
    today = datetime.now().date().strftime('%Y-%m-%d')
    if today not in st.session_state.water_log:
        st.session_state.water_log[today] = []
    st.session_state.water_log[today].append({
        'time': datetime.now().strftime('%H:%M'),
        'amount': amount
    })
    st.success(f"Added {amount} ml of water!")
    st.rerun()
