import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import ast
import re

from constants import DEFAULT_TARGETS

from utils import (
    load_db, search_food_case_insensitive, convert_units_to_grams,
    get_display_amount_and_unit, calculate_nutrition_per_serving,
    get_meal_totals, get_progress_towards_targets, aggregate_historical_data,
    aggregate_meal_wise_data, calculate_bmr_tdee, calculate_macros,
    filter_foods_by_preferences, generate_smart_swaps, filter_meals,
    build_week, build_day, create_macro_chart, create_micronutrient_chart,
    get_random_tips, calculate_adaptive_recommendations, calculate_bmi,
    create_daily_chart, create_weekly_chart, create_monthly_chart,
    create_meal_wise_chart, create_daily_calorie_bar_chart,
    create_meal_wise_macro_bar_chart, calculate_ideal_body_weight,
    calculate_body_fat_percentage, calculate_lean_body_mass,
    calculate_protein_target, calculate_daily_water_intake,
    calculate_calorie_balance, calculate_macro_ratios_string
)
from utils.analytics import (
    create_weekly_nutrient_bar_chart, create_daily_calorie_pie_chart,
    create_protein_line_plot, calculate_weekly_summary_stats
)

# Global variables
meal_tracking_df = None

# Page configuration
st.set_page_config(
    page_title="Nutrition & Fitness Planner",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    """Load all necessary data files."""
    global meal_tracking_df

    try:
        nutrition_df = load_db()
    except FileNotFoundError:
        st.error("Nutrition database not found. Please ensure 'data/nutrition_data_optimized.csv' exists.")
        nutrition_df = pd.DataFrame()

    try:
        tips_df = pd.read_csv('data/tips.csv')
    except FileNotFoundError:
        tips_df = pd.DataFrame()

    try:
        daily_meals_df = pd.read_csv('data/daily_meals.csv')
    except FileNotFoundError:
        daily_meals_df = pd.DataFrame()

    try:
        meal_tracking_df = pd.read_csv('data/meal_tracking.csv')
    except FileNotFoundError:
        meal_tracking_df = pd.DataFrame()

    return nutrition_df, tips_df, daily_meals_df

# Initialize session state
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = None
if 'meal_plan' not in st.session_state:
    st.session_state.meal_plan = None

if 'metrics_calculated' not in st.session_state:
    st.session_state.metrics_calculated = False
if 'daily_log' not in st.session_state:
    st.session_state.daily_log = {}
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'edit_meal_index' not in st.session_state:
    st.session_state.edit_meal_index = None
if 'edit_category' not in st.session_state:
    st.session_state.edit_category = None
if 'show_plan_types' not in st.session_state:
    st.session_state.show_plan_types = False

# Load data
nutrition_df, tips_df, daily_meals_df = load_data()

# Custom CSS
st.markdown("""
<style>
    .meal-plan-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        border: 2px solid #388E3C;
    }
    .meal-plan-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .meal-item {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        color: #2E7D32;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
    }
    .meal-item strong {
        color: #1B5E20;
        font-size: 1.1em;
    }
    .metric-card {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border: 2px solid #0D47A1;
    }
    .tab-content {
        padding: 25px;
        background: #f5f5f5;
        border-radius: 15px;
        margin: 15px 0;
        border: 1px solid #e0e0e0;
    }
    .progress-bar {
        height: 25px;
        border-radius: 12px;
        background: #e0e0e0;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #66BB6A);
        transition: width 0.3s ease;
    }
    .smart-swap-btn {
        background: #FF9800;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .smart-swap-btn:hover {
        background: #F57C00;
    }
    .plan-option-card {
        background: linear-gradient(135deg, #9C27B0 0%, #7B1FA2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #6A1B9A;
    }
    .plan-option-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    @media (max-width: 768px) {
        .meal-plan-metrics {
            flex-direction: column;
        }
        .meal-plan-card, .metric-card {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Nutrition & Fitness Planner</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Personal Information")

    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    activity_level = st.selectbox("Activity Level",
                                  ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])

    goal = st.selectbox("Goal", ["Lose weight", "Maintain", "Gain muscle"])

    st.header("Dietary Preferences")
    food_preference = st.selectbox("Food Preference", ["None", "Vegetarian", "Vegan"])
    allergies = st.text_input("Allergies (comma-separated)", "")

    if st.button("Calculate My Needs"):
        try:
            bmr, tdee = calculate_bmr_tdee(age, weight, height, gender, activity_level)
            adjusted_calories = tdee

            if goal == "Lose weight":
                adjusted_calories -= 500
            elif goal == "Gain muscle":
                adjusted_calories += 500

            protein_pct, carbs_pct, fat_pct, protein_g, carbs_g, fat_g = calculate_macros(adjusted_calories, goal)

            # Additional calculations
            ideal_weight = calculate_ideal_body_weight(height, gender)
            body_fat_pct = calculate_body_fat_percentage(weight, height, age, gender)
            lean_body_mass = calculate_lean_body_mass(weight, body_fat_pct)
            protein_target = calculate_protein_target(weight, goal)
            water_intake = calculate_daily_water_intake(weight, activity_level)

            # Calculate calorie balance if meal plan exists
            calorie_balance = 0
            if st.session_state.meal_plan:
                consumed_calories = sum(meal['Calories'] for day_meals in st.session_state.meal_plan.values() for meal in day_meals)
                calorie_balance = calculate_calorie_balance(consumed_calories, adjusted_calories)

            # Store in session state
            st.session_state.adjusted_calories = adjusted_calories
            st.session_state.protein_g = protein_g
            st.session_state.carbs_g = carbs_g
            st.session_state.fat_g = fat_g
            st.session_state.bmr = bmr
            st.session_state.tdee = tdee
            st.session_state.ideal_weight = ideal_weight
            st.session_state.body_fat_pct = body_fat_pct
            st.session_state.lean_body_mass = lean_body_mass
            st.session_state.protein_target = protein_target
            st.session_state.water_intake = water_intake
            st.session_state.calorie_balance = calorie_balance
            st.session_state.metrics_calculated = True

            bmi, bmi_category = calculate_bmi(weight, height)
            st.session_state.bmi = bmi
            st.session_state.bmi_category = bmi_category

        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")

from water_tracker import water_tracker_tab
from weight_tracker import weight_tracker_tab
from streaks import streaks_tab

# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["BMI & Daily Calorie Needs", "Meal Plan Generator", "Health Metrics", "Meal Tracking", "Water", "Weight", "Streaks"])

with tab1:
    st.header("Meals")

    # BMI & Calorie Needs Section
    st.subheader("BMI & Daily Calorie Needs")
    if st.session_state.metrics_calculated:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("BMI", f"{st.session_state.bmi:.1f}", st.session_state.bmi_category)

        with col2:
            st.metric("BMR", f"{st.session_state.bmr:.0f} kcal/day")

        with col3:
            st.metric("TDEE", f"{st.session_state.tdee:.0f} kcal/day")

        st.metric("Adjusted Daily Calories", f"{st.session_state.adjusted_calories:.0f} kcal")

        # Macro breakdown
        st.subheader("Macronutrient Targets")
        macro_col1, macro_col2, macro_col3 = st.columns(3)

        with macro_col1:
            st.metric("Protein", f"{st.session_state.protein_g:.0f}g")

        with macro_col2:
            st.metric("Carbs", f"{st.session_state.carbs_g:.0f}g")

        with macro_col3:
            st.metric("Fat", f"{st.session_state.fat_g:.0f}g")

    else:
        st.info("Please enter your information in the sidebar and click 'Calculate My Needs' to see your metrics.")



with tab2:
    st.header("Meal Plan Generator")

    if st.session_state.metrics_calculated:
        if not st.session_state.show_plan_types:
            if st.button("Generate Meal Plan"):
                st.session_state.show_plan_types = True
                st.rerun()
        else:
            plan_type = st.radio("Select Plan Type", ["Balanced Plan", "High-Protein Plan", "Weight-Loss Plan"])

            if st.button("Generate Selected Plan"):
                with st.spinner("Generating meal plan..."):
                    try:
                        # Map plan type to goal
                        if plan_type == "Balanced Plan":
                            temp_goal = "Maintain"
                        elif plan_type == "High-Protein Plan":
                            temp_goal = "Gain muscle"
                        else:
                            temp_goal = "Lose weight"

                        # Recalculate metrics based on selected plan type
                        bmr, tdee = calculate_bmr_tdee(age, weight, height, gender, activity_level)
                        adjusted_calories = tdee

                        if temp_goal == "Lose weight":
                            adjusted_calories -= 500
                        elif temp_goal == "Gain muscle":
                            adjusted_calories += 500

                        protein_pct, carbs_pct, fat_pct, protein_g, carbs_g, fat_g = calculate_macros(adjusted_calories, temp_goal)

                        # Filter foods based on preferences
                        filtered_df = filter_foods_by_preferences(
                            nutrition_df, food_preference, allergies
                        )

                        if filtered_df.empty:
                            st.error("No foods match your preferences. Please adjust your dietary preferences.")
                        else:
                            # Build meal plan
                            daily_targets = {
                                'calories': adjusted_calories,
                                'protein': protein_g,
                                'carbs': carbs_g,
                                'fat': fat_g
                            }

                            meal_plan_result = build_week(filtered_df, daily_targets)
                            st.session_state.meal_plan = meal_plan_result['week_plan']
                            st.session_state.show_plan_types = False

                            st.success("Meal plan generated successfully!")

                    except Exception as e:
                        st.error(f"Error generating meal plan: {str(e)}")

        if st.session_state.meal_plan:
            st.subheader("Your 7-Day Meal Plan")

            # Weekly totals
            weekly_totals = {'Calories': 0, 'Protein': 0, 'Carbs': 0, 'Fat': 0}
            for day_key, day_meals in st.session_state.meal_plan.items():
                day_totals = get_meal_totals(day_meals)
                weekly_totals['Calories'] += day_totals['Calories']
                weekly_totals['Protein'] += day_totals['Protein']
                weekly_totals['Carbs'] += day_totals['Carbs']
                weekly_totals['Fat'] += day_totals['Fat']

            st.subheader("Weekly Summary")
            weekly_col1, weekly_col2, weekly_col3, weekly_col4 = st.columns(4)
            with weekly_col1:
                st.metric("Total Calories", f"{weekly_totals['Calories']:.0f}")
            with weekly_col2:
                st.metric("Total Protein", f"{weekly_totals['Protein']:.1f}g")
            with weekly_col3:
                st.metric("Total Carbs", f"{weekly_totals['Carbs']:.1f}g")
            with weekly_col4:
                st.metric("Total Fat", f"{weekly_totals['Fat']:.1f}g")

            # Display each day
            for day_key, day_meals in st.session_state.meal_plan.items():
                day_num = day_key.split('_')[1]
                st.markdown(f"### Day {day_num}")

                # Group meals by type
                meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
                for meal_type in meal_types:
                    meals_of_type = [meal for meal in day_meals if meal['Meal'] == meal_type]
                    if meals_of_type:
                        st.markdown(f"""
                        <div class="meal-plan-card">
                            <h4>{meal_type}</h4>
                        </div>
                        """, unsafe_allow_html=True)

                        for meal in meals_of_type:
                            # Generate smart swaps for this food
                            smart_swaps = generate_smart_swaps(meal['Food'], nutrition_df)

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"""
                                <div class="meal-item">
                                    <strong>{meal['Food']}</strong> - {meal['Amount']} |
                                    <span style="color: #007bff;">{meal['Calories']:.0f} kcal</span> |
                                    <span style="color: #28a745;">{meal['Protein']:.1f}g protein</span> |
                                    <span style="color: #ffc107;">{meal['Carbs']:.1f}g carbs</span> |
                                    <span style="color: #dc3545;">{meal['Fat']:.1f}g fat</span>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                if smart_swaps:
                                    if st.button("🔄", key=f"swap_{meal['Food']}_{day_key}_{meal_type}"):
                                        import random
                                        swap_food_dict = random.choice(smart_swaps)
                                        swap_food_name = swap_food_dict['food']
                                        # Parse quantity and unit
                                        amount_parts = meal['Amount'].split()
                                        quantity_str = amount_parts[0]
                                        # Extract numeric value using regex
                                        match = re.match(r'(\d+\.?\d*)', quantity_str)
                                        if match:
                                            quantity = float(match.group(1))
                                        else:
                                            quantity = 1.0  # fallback
                                        unit = ' '.join(amount_parts[1:])
                                        # Recalculate nutrition for swap
                                        try:
                                            new_nutrition = calculate_nutrition_per_serving(swap_food_name, quantity, unit, nutrition_df)
                                            new_display_amount, new_display_unit = get_display_amount_and_unit(swap_food_name, new_nutrition['Calories'] / (nutrition_df.loc[swap_food_name, 'kcal'] / 100), nutrition_df)
                                            # Find and replace the meal
                                            for i, m in enumerate(st.session_state.meal_plan[day_key]):
                                                if m['Food'] == meal['Food'] and m['Meal'] == meal_type:
                                                    st.session_state.meal_plan[day_key][i] = {
                                                        'Food': swap_food_name,
                                                        'Amount': f"{new_display_amount} {new_display_unit}",
                                                        'Calories': new_nutrition['Calories'],
                                                        'Protein': new_nutrition['Protein'],
                                                        'Carbs': new_nutrition['Carbs'],
                                                        'Fat': new_nutrition['Fat'],
                                                        'Meal': meal_type
                                                    }
                                                    break
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error swapping food: {str(e)}")

                # Display daily totals
                day_totals = get_meal_totals(day_meals)
                st.subheader(f"Day {day_num} Totals")
                total_col1, total_col2, total_col3, total_col4 = st.columns(4)

                with total_col1:
                    st.metric("Calories", f"{day_totals['Calories']:.0f}")
                with total_col2:
                    st.metric("Protein", f"{day_totals['Protein']:.1f}g")
                with total_col3:
                    st.metric("Carbs", f"{day_totals['Carbs']:.1f}g")
                with total_col4:
                    st.metric("Fat", f"{day_totals['Fat']:.1f}g")

                st.markdown("---")

            # Enhanced Visualizations
            st.header("📊 Meal Plan Analytics")

            # Weekly macro distribution bar plot
            weekly_bar_chart = create_weekly_nutrient_bar_chart(st.session_state.meal_plan)
            if weekly_bar_chart:
                st.subheader("Weekly Nutrient Distribution")
                st.plotly_chart(weekly_bar_chart, use_container_width=True)

            # Daily calorie trend line plot
            protein_line_chart = create_protein_line_plot(st.session_state.meal_plan)
            if protein_line_chart:
                st.subheader("Weekly Protein Intake Trend")
                st.plotly_chart(protein_line_chart, use_container_width=True)

            # Macro ratio pie chart for each day
            st.subheader("Daily Macro Ratios")
            day_cols = st.columns(7)

            for i, (day_key, day_meals) in enumerate(st.session_state.meal_plan.items()):
                with day_cols[i]:
                    day_name = day_key.replace('_', ' ').title()
                    pie_chart = create_daily_calorie_pie_chart(st.session_state.meal_plan, day_key)
                    if pie_chart:
                        st.markdown(f"**{day_name}**")
                        st.plotly_chart(pie_chart, use_container_width=True)

    else:
        st.info("Please calculate your needs first in the sidebar.")

with tab3:
    st.header("Health Metrics")

    if st.session_state.metrics_calculated:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Ideal Body Weight", f"{st.session_state.ideal_weight:.1f} kg")
            st.metric("Body Fat Percentage", f"{st.session_state.body_fat_pct:.1f}%")
            st.metric("Lean Body Mass", f"{st.session_state.lean_body_mass:.1f} kg")

        with col2:
            st.metric("Protein Target", f"{st.session_state.protein_target:.1f}g")
            st.metric("Daily Water Intake", f"{st.session_state.water_intake:.0f} ml")
            if st.session_state.meal_plan:
                st.metric("Calorie Balance", f"{st.session_state.calorie_balance:.0f} kcal")
            else:
                st.metric("Calorie Balance", "N/A (Generate meal plan first)")

        # Additional metrics
        st.subheader("Body Composition Insights")
        if st.session_state.body_fat_pct < 10:
            st.info("Your body fat percentage is low. Consider maintaining or increasing calorie intake if you're active.")
        elif st.session_state.body_fat_pct > 25:
            st.info("Your body fat percentage is high. Focus on calorie deficit and exercise for weight loss.")
        else:
            st.success("Your body fat percentage is in a healthy range.")

    else:
        st.info("Please calculate your needs in the sidebar to see health metrics.")

with tab4:
    st.header("Meal Tracking")

    # Date picker
    selected_date = st.date_input("Select Date", datetime.now().date())

    # Meal type selector
    meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

    # Food selector
    if not nutrition_df.empty:
        food_options = [""] + nutrition_df.index.tolist()
        selected_food = st.selectbox("Select Food", food_options)

        # Quantity input
        quantity = st.number_input("Quantity", min_value=0.1, value=1.0)

        # Unit selector
        unit_options = ["grams", "cups", "pieces", "fruit sizes", "sabzi types"]
        unit = st.selectbox("Unit", unit_options)

        if st.button("Add to Meal"):
            if selected_food:
                try:
                    nutrition = calculate_nutrition_per_serving(selected_food, quantity, unit, nutrition_df)
                    display_amount, display_unit = get_display_amount_and_unit(selected_food, nutrition['Calories'] / (nutrition_df.loc[selected_food, 'kcal'] / 100), nutrition_df)

                    meal_entry = {
                        'Date': selected_date.strftime('%Y-%m-%d'),
                        'Meal_Type': meal_type,
                        'Food': selected_food,
                        'Amount': f"{display_amount} {display_unit}",
                        'Calories': nutrition['Calories'],
                        'Protein': nutrition['Protein'],
                        'Carbs': nutrition['Carbs'],
                        'Fat': nutrition['Fat']
                    }

                    # Add to daily log
                    date_str = selected_date.strftime('%Y-%m-%d')
                    if date_str not in st.session_state.daily_log:
                        st.session_state.daily_log[date_str] = []

                    st.session_state.daily_log[date_str].append(meal_entry)
                    st.success(f"Added {selected_food} to {meal_type} for {selected_date.strftime('%B %d, %Y')}")

                except Exception as e:
                    st.error(f"Error adding meal: {str(e)}")
            else:
                st.warning("Please select a food item.")

    # Display today's meals
    if selected_date.strftime('%Y-%m-%d') in st.session_state.daily_log:
        st.subheader(f"Meals for {selected_date.strftime('%B %d, %Y')}")

        meals_today = st.session_state.daily_log[selected_date.strftime('%Y-%m-%d')]
        if meals_today:
            df_today = pd.DataFrame(meals_today)
            st.dataframe(df_today, use_container_width=True)

            # Calculate totals
            totals_today = get_meal_totals(meals_today)
            st.subheader("Daily Totals")
            total_today_col1, total_today_col2, total_today_col3, total_today_col4 = st.columns(4)

            with total_today_col1:
                st.metric("Calories", f"{totals_today['Calories']:.0f}")
            with total_today_col2:
                st.metric("Protein", f"{totals_today['Protein']:.1f}g")
            with total_today_col3:
                st.metric("Carbs", f"{totals_today['Carbs']:.1f}g")
            with total_today_col4:
                st.metric("Fat", f"{totals_today['Fat']:.1f}g")

            # Progress towards targets
            if st.session_state.metrics_calculated:
                progress = get_progress_towards_targets(totals_today, {
                    'Calories': st.session_state.adjusted_calories,
                    'Protein': st.session_state.protein_g,
                    'Carbs': st.session_state.carbs_g,
                    'Fat': st.session_state.fat_g
                })

                st.subheader("Progress Towards Targets")
                progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)

                for i, macro in enumerate(['Calories', 'Protein', 'Carbs', 'Fat']):
                    with [progress_col1, progress_col2, progress_col3, progress_col4][i]:
                        pct = progress[macro]['percentage']
                        st.metric(f"{macro} Progress", f"{pct:.0f}%",
                                 f"{progress[macro]['remaining']:.0f} remaining")
                        st.progress(min(pct / 100, 1.0))
        else:
            st.info("No meals logged for this date.")
    else:
        st.info("No meals logged for this date.")

    # Analytics section
    if st.session_state.daily_log:
        st.header("Analytics")

        # Aggregate data
        historical_data = aggregate_historical_data(st.session_state.daily_log)
        meal_data = aggregate_meal_wise_data(st.session_state.daily_log)

        if historical_data:
            analytics_col1, analytics_col2 = st.columns(2)

            with analytics_col1:
                st.subheader("Daily Intake Trends")
                macro_select = st.selectbox("Select Macro", ["Calories", "Protein", "Carbs", "Fat"], key="daily_trends")
                daily_chart = create_daily_chart(historical_data, macro_select)
                if daily_chart:
                    st.plotly_chart(daily_chart, use_container_width=True)

            with analytics_col2:
                st.subheader("Weekly Summary")
                weekly_chart = create_weekly_chart(historical_data, macro_select)
                if weekly_chart:
                    st.plotly_chart(weekly_chart, use_container_width=True)

            # Meal-wise breakdown
            st.subheader("Meal-wise Breakdown")
            period_select = st.selectbox("Period", ["daily", "weekly", "monthly"], key="meal_wise_period")
            meal_wise_chart = create_meal_wise_chart(meal_data, macro_select, period_select)
            if meal_wise_chart:
                st.plotly_chart(meal_wise_chart, use_container_width=True)

with tab5:
    water_tracker_tab()

with tab6:
    weight_tracker_tab()

with tab7:
    streaks_tab()


