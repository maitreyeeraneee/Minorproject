import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar

def streaks_tab():
    st.header("🔥 Streaks")

    if not st.session_state.daily_log:
        st.info("Start logging meals to build your streaks!")
        return

    # Calculate streaks
    current_streak, longest_streak, streak_data = calculate_streaks()

    # Display streak metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Streak", f"{current_streak} days")
    with col2:
        st.metric("Longest Streak", f"{longest_streak} days")
    with col3:
        total_logged_days = len(streak_data)
        st.metric("Total Logged Days", total_logged_days)

    # Streak heatmap
    st.subheader("Meal Logging Heatmap")
    heatmap_fig = create_streak_heatmap(streak_data)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Streak history
    st.subheader("Streak History")
    if streak_data:
        df = pd.DataFrame(list(streak_data.items()), columns=['Date', 'Logged'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)

        # Show recent activity
        recent_df = df.head(30)  # Last 30 days
        for _, row in recent_df.iterrows():
            status = "✅ Logged" if row['Logged'] else "❌ Missed"
            st.write(f"{row['Date'].strftime('%b %d, %Y')}: {status}")

def calculate_streaks():
    if not st.session_state.daily_log:
        return 0, 0, {}

    # Get all dates with logging
    logged_dates = set()
    for date_str, meals in st.session_state.daily_log.items():
        if meals:  # Has meals logged
            logged_dates.add(date_str)

    # Create date range from first log to today
    if not logged_dates:
        return 0, 0, {}

    start_date = min(logged_dates)
    end_date = datetime.now().date()

    # Build streak data
    streak_data = {}
    current_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_datetime = datetime.combine(end_date, datetime.min.time())

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        streak_data[date_str] = date_str in logged_dates
        current_date += timedelta(days=1)

    # Calculate current streak
    current_streak = 0
    check_date = end_date
    while check_date >= datetime.strptime(start_date, '%Y-%m-%d').date():
        date_str = check_date.strftime('%Y-%m-%d')
        if date_str in streak_data and streak_data[date_str]:
            current_streak += 1
            check_date -= timedelta(days=1)
        else:
            break

    # Calculate longest streak
    longest_streak = 0
    temp_streak = 0
    for date_str in sorted(streak_data.keys()):
        if streak_data[date_str]:
            temp_streak += 1
            longest_streak = max(longest_streak, temp_streak)
        else:
            temp_streak = 0

    return current_streak, longest_streak, streak_data

def create_streak_heatmap(streak_data):
    if not streak_data:
        return None

    # Prepare data for heatmap
    data = []
    for date_str, logged in streak_data.items():
        date = datetime.strptime(date_str, '%Y-%m-%d')
        data.append({
            'date': date,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'week': date.isocalendar()[1],
            'weekday': date.weekday(),
            'logged': 1 if logged else 0
        })

    df = pd.DataFrame(data)

    # Create heatmap
    fig = go.Figure()

    # Group by week and weekday
    pivot = df.pivot_table(values='logged', index='weekday', columns='week', aggfunc='sum', fill_value=0)

    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale=[
            [0, '#ebedf0'],  # No activity
            [0.5, '#9be9a8'],  # Some activity
            [1, '#40c463']  # Full activity
        ],
        showscale=False,
        hoverongaps=False
    ))

    fig.update_layout(
        title="Meal Logging Activity",
        xaxis_title="Week",
        yaxis_title="Day",
        height=200
    )

    return fig
