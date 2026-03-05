try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Conditional decorator for streamlit caching
def cache_data(func):
    if HAS_STREAMLIT:
        return st.cache_data(func)
    else:
        return func

def calculate_bmi(weight, height):
    """
    Calculate BMI and category.
    """
    bmi = round(weight / ((height/100)**2), 1)
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Normal"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return bmi, category

@cache_data
def create_macro_chart(protein_g, carbs_g, fat_g):
    """
    Create a pie chart for macronutrient distribution.
    """
    if HAS_STREAMLIT:
        values = [protein_g, carbs_g, fat_g]
        labels = ['Protein', 'Carbs', 'Fat']
        fig = px.pie(values=values, names=labels, title="Macronutrient Distribution")
        return fig
    else:
        return None

@cache_data
def create_micronutrient_chart():
    """
    Placeholder for micronutrient chart.
    """
    if HAS_STREAMLIT:
        fig = px.bar(title="Micronutrient Intake - Data Not Available")
        return fig
    else:
        return None

def get_random_tips(tips_df, n=3):
    """
    Get random tips from tips dataframe.
    """
    if len(tips_df) > 0:
        return tips_df.sample(min(n, len(tips_df)))['tip'].tolist()
    return []

def calculate_adaptive_recommendations(user_feedback, current_plan):
    """
    Generate adaptive recommendations based on feedback.
    """
    return "Based on your feedback, we'll adjust future meal plans to better suit your preferences."

@cache_data
def create_daily_chart(daily_data, macro='Calories'):
    """
    Create a line chart for daily intake over time.
    """
    if not daily_data:
        return px.line(title=f"Daily {macro} Intake - No Data Available")

    df = pd.DataFrame(daily_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    fig = px.line(df, x='Date', y=macro, title=f"Daily {macro} Intake",
                  markers=True, line_shape='linear')
    fig.update_layout(xaxis_title="Date", yaxis_title=f"{macro}")
    return fig

@cache_data
def create_weekly_chart(daily_data, macro='Calories'):
    """
    Create a bar chart for weekly aggregated intake.
    """
    if not daily_data:
        return px.bar(title=f"Weekly {macro} Intake - No Data Available")

    df = pd.DataFrame(daily_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.to_period('W').astype(str)
    weekly_data = df.groupby('Week')[macro].sum().reset_index()

    fig = px.bar(weekly_data, x='Week', y=macro, title=f"Weekly {macro} Intake")
    fig.update_layout(xaxis_title="Week", yaxis_title=f"{macro}")
    return fig

@cache_data
def create_monthly_chart(daily_data, macro='Calories'):
    """
    Create a bar chart for monthly aggregated intake.
    """
    if not daily_data:
        return px.bar(title=f"Monthly {macro} Intake - No Data Available")

    df = pd.DataFrame(daily_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly_data = df.groupby('Month')[macro].sum().reset_index()

    fig = px.bar(monthly_data, x='Month', y=macro, title=f"Monthly {macro} Intake")
    fig.update_layout(xaxis_title="Month", yaxis_title=f"{macro}")
    return fig

@cache_data
def create_meal_wise_chart(meal_data, macro='Calories', period='daily'):
    """
    Create a stacked bar chart showing meal-wise breakdown.
    meal_data should be a list of dicts with Date, Meal_Type, and macro values.
    """
    if not meal_data:
        return px.bar(title=f"Meal-wise {macro} Intake - No Data Available")

    df = pd.DataFrame(meal_data)
    df['Date'] = pd.to_datetime(df['Date'])

    if period == 'weekly':
        df['Period'] = df['Date'].dt.to_period('W').astype(str)
    elif period == 'monthly':
        df['Period'] = df['Date'].dt.to_period('M').astype(str)
    else:  # daily
        df['Period'] = df['Date'].dt.date.astype(str)

    # Ensure Meal_Type column exists
    if 'Meal_Type' not in df.columns:
        df['Meal_Type'] = 'Unknown'

    # Group by period and meal type
    grouped_data = df.groupby(['Period', 'Meal_Type'])[macro].sum().reset_index()

    fig = px.bar(grouped_data, x='Period', y=macro, color='Meal_Type',
                 title=f"Meal-wise {macro} Intake ({period.capitalize()})",
                 barmode='stack')
    fig.update_layout(xaxis_title="Period", yaxis_title=f"{macro}")
    return fig

@cache_data
def create_daily_calorie_bar_chart(meal_plan):
    """
    Create a bar chart showing daily calorie distribution for a meal plan.
    """
    if not meal_plan:
        return px.bar(title="Daily Calorie Distribution - No Meal Plan Selected")

    # Group meals by type and sum calories
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    calories = []

    for meal_type in meal_types:
        meal_calories = sum(item['Calories'] for item in meal_plan if item['Meal'] == meal_type)
        calories.append(meal_calories)

    fig = px.bar(x=meal_types, y=calories, title="Daily Calorie Distribution by Meal",
                 labels={'x': 'Meal Type', 'y': 'Calories'},
                 color=meal_types,
                 color_discrete_sequence=['#FF9999', '#66B3FF', '#99FF99', '#FFD700'])
    fig.update_layout(showlegend=False)
    return fig

@cache_data
def create_meal_wise_macro_bar_chart(meal_plan):
    """
    Create a grouped bar chart showing macros for each meal type.
    """
    if not meal_plan:
        return px.bar(title="Meal-wise Macro Distribution - No Meal Plan Selected")

    # Group by meal type and sum macros
    meal_data = {}
    for item in meal_plan:
        meal_type = item['Meal']
        if meal_type not in meal_data:
            meal_data[meal_type] = {'Protein': 0, 'Carbs': 0, 'Fat': 0}
        meal_data[meal_type]['Protein'] += item['Protein']
        meal_data[meal_type]['Carbs'] += item['Carbs']
        meal_data[meal_type]['Fat'] += item['Fat']

    # Prepare data for plotting
    meal_types = list(meal_data.keys())
    protein_vals = [meal_data[mt]['Protein'] for mt in meal_types]
    carbs_vals = [meal_data[mt]['Carbs'] for mt in meal_types]
    fat_vals = [meal_data[mt]['Fat'] for mt in meal_types]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Protein', x=meal_types, y=protein_vals, marker_color='#FF9999'))
    fig.add_trace(go.Bar(name='Carbs', x=meal_types, y=carbs_vals, marker_color='#66B3FF'))
    fig.add_trace(go.Bar(name='Fat', x=meal_types, y=fat_vals, marker_color='#99FF99'))

    fig.update_layout(
        title="Meal-wise Macronutrient Distribution",
        xaxis_title="Meal Type",
        yaxis_title="Grams",
        barmode='group'
    )
    return fig
