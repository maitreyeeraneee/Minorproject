def calculate_ideal_body_weight(height, gender):
    """
    Calculate ideal body weight using Devine formula.
    """
    if gender.lower() == 'male':
        ibw = 50 + 2.3 * (height - 152.4) / 2.54
    else:
        ibw = 45.5 + 2.3 * (height - 152.4) / 2.54
    return round(ibw, 1)

def calculate_body_fat_percentage(weight, height, age, gender):
    """
    Estimate body fat percentage using BMI method.
    """
    bmi = weight / ((height/100)**2)
    if gender.lower() == 'male':
        bfp = (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        bfp = (1.20 * bmi) + (0.23 * age) - 5.4
    return round(max(0, min(50, bfp)), 1)

def calculate_lean_body_mass(weight, body_fat_pct):
    """
    Calculate lean body mass.
    """
    return round(weight * (1 - body_fat_pct/100), 1)

def calculate_protein_target(weight, goal):
    """
    Calculate daily protein target.
    """
    if goal.lower() == 'gain muscle':
        protein_per_kg = 2.0
    elif goal.lower() == 'lose weight':
        protein_per_kg = 1.6
    else:
        protein_per_kg = 1.2
    return round(weight * protein_per_kg, 1)

def calculate_daily_water_intake(weight, activity_level):
    """
    Calculate daily water intake recommendation.
    """
    base_intake = weight * 30  # 30ml/kg
    if activity_level.lower() in ['moderately active', 'very active', 'extra active']:
        base_intake *= 1.2
    return round(base_intake, 0)

def calculate_calorie_balance(intake, needs):
    """
    Calculate calorie balance (intake - needs).
    """
    return round(intake - needs, 0)

def calculate_macro_ratios_string(protein_pct, carbs_pct, fat_pct):
    """
    Format macro ratios as a string.
    """
    return f"Protein: {protein_pct*100:.0f}%, Carbs: {carbs_pct*100:.0f}%, Fat: {fat_pct*100:.0f}%"

def calculate_weekly_summary_stats(meal_plan):
    """
    Calculate weekly summary statistics from meal plan data.

    Args:
        meal_plan: Dict with day keys containing lists of meal items

    Returns:
        Dict with weekly statistics
    """
    if not meal_plan:
        return {
            'avg_daily_calories': 0,
            'total_weekly_calories': 0,
            'avg_daily_protein': 0,
            'protein_consistency': 0
        }

    total_calories = 0
    total_protein = 0
    daily_proteins = []
    day_count = len(meal_plan)

    for day_meals in meal_plan.values():
        day_calories = sum(meal.get('Calories', 0) for meal in day_meals)
        day_protein = sum(meal.get('Protein', 0) for meal in day_meals)

        total_calories += day_calories
        total_protein += day_protein
        daily_proteins.append(day_protein)

    # Calculate averages
    avg_daily_calories = total_calories / day_count if day_count > 0 else 0
    avg_daily_protein = total_protein / day_count if day_count > 0 else 0

    # Calculate protein consistency (coefficient of variation)
    if len(daily_proteins) > 1 and avg_daily_protein > 0:
        protein_std = sum((p - avg_daily_protein) ** 2 for p in daily_proteins) / len(daily_proteins)
        protein_std = protein_std ** 0.5  # Standard deviation
        protein_consistency = protein_std / avg_daily_protein
    else:
        protein_consistency = 0

    return {
        'avg_daily_calories': round(avg_daily_calories, 1),
        'total_weekly_calories': round(total_calories, 1),
        'avg_daily_protein': round(avg_daily_protein, 1),
        'protein_consistency': round(protein_consistency, 2)
    }

def create_weekly_nutrient_bar_chart(meal_plan):
    """
    Create a grouped bar chart showing daily nutrient totals.

    Args:
        meal_plan: Dict with day keys containing lists of meal items

    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not meal_plan:
        fig = go.Figure()
        fig.update_layout(title="No Meal Plan Data Available")
        return fig

    days = []
    calories = []
    proteins = []
    carbs = []
    fats = []

    for day_key, day_meals in meal_plan.items():
        day_name = day_key.replace('_', ' ').title()
        days.append(day_name)

        day_calories = sum(meal.get('Calories', 0) for meal in day_meals)
        day_protein = sum(meal.get('Protein', 0) for meal in day_meals)
        day_carbs = sum(meal.get('Carbs', 0) for meal in day_meals)
        day_fat = sum(meal.get('Fat', 0) for meal in day_meals)

        calories.append(day_calories)
        proteins.append(day_protein)
        carbs.append(day_carbs)
        fats.append(day_fat)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Calories', x=days, y=calories, marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(name='Protein', x=days, y=proteins, marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='Carbs', x=days, y=carbs, marker_color='#45B7D1'))
    fig.add_trace(go.Bar(name='Fat', x=days, y=fats, marker_color='#FFA07A'))

    fig.update_layout(
        title="Weekly Nutrient Totals",
        xaxis_title="Day",
        yaxis_title="Amount",
        barmode='group',
        showlegend=True
    )

    return fig

def create_daily_calorie_pie_chart(meal_plan, day='day_1'):
    """
    Create a pie chart showing calorie distribution by meal type for a specific day.

    Args:
        meal_plan: Dict with day keys containing lists of meal items
        day: Day key to analyze (default: 'day_1')

    Returns:
        Plotly figure object
    """
    try:
        import plotly.express as px
    except ImportError:
        return None

    if not meal_plan or day not in meal_plan:
        fig = px.pie(title=f"No data available for {day}")
        return fig

    day_meals = meal_plan[day]
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    calories = []

    for meal_type in meal_types:
        meal_calories = sum(meal.get('Calories', 0) for meal in day_meals if meal.get('Meal') == meal_type)
        calories.append(meal_calories)

    # Filter out zero values for cleaner chart
    filtered_data = [(mt, cal) for mt, cal in zip(meal_types, calories) if cal > 0]
    if not filtered_data:
        fig = px.pie(title=f"No calorie data for {day}")
        return fig

    meal_types_filtered, calories_filtered = zip(*filtered_data)

    fig = px.pie(
        values=calories_filtered,
        names=meal_types_filtered,
        title=f"Calorie Distribution - {day.replace('_', ' ').title()}",
        color_discrete_sequence=['#FF9999', '#66B3FF', '#99FF99', '#FFD700']
    )

    return fig

def create_protein_line_plot(meal_plan):
    """
    Create a line plot showing protein intake trends across the week.

    Args:
        meal_plan: Dict with day keys containing lists of meal items

    Returns:
        Plotly figure object
    """
    try:
        import plotly.express as px
    except ImportError:
        return None

    if not meal_plan:
        fig = px.line(title="No Meal Plan Data Available")
        return fig

    days = []
    protein_totals = []

    for day_key, day_meals in meal_plan.items():
        day_name = day_key.replace('_', ' ').title()
        days.append(day_name)

        day_protein = sum(meal.get('Protein', 0) for meal in day_meals)
        protein_totals.append(day_protein)

    fig = px.line(
        x=days,
        y=protein_totals,
        title="Weekly Protein Intake Trend",
        markers=True,
        line_shape='linear'
    )

    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Protein (g)",
        showlegend=False
    )

    fig.update_traces(line_color='#4ECDC4', marker_color='#4ECDC4')

    return fig
