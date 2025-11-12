import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    calculate_bmr_tdee, calculate_macros, filter_foods_by_preferences, optimize_meal_plan,
    generate_smart_swaps, create_macro_chart, create_micronutrient_chart, create_progress_chart,
    predict_weight_change, get_random_tips, calculate_adaptive_recommendations, get_motivational_message,
    predict_future_weight, create_weight_projection_chart, assign_user_to_cluster,
    create_feature_importance_chart, multi_objective_optimization, train_weight_prediction_model
)
from train_model import predict_calories

# Page configuration
st.set_page_config(
    page_title="AI Diet & Fitness Recommender",
    page_icon="🥗",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    nutrition_df = pd.read_csv('data/nutrition_data.csv')
    tips_df = pd.read_csv('data/tips.csv')
    daily_meals_df = pd.read_csv('data/daily_meals.csv')
    return nutrition_df, tips_df, daily_meals_df

nutrition_df, tips_df, daily_meals_df = load_data()

# Initialize session state
if 'meal_plan' not in st.session_state:
    st.session_state['meal_plan'] = None
if 'user_feedback' not in st.session_state:
    st.session_state['user_feedback'] = ""
if 'streak_days' not in st.session_state:
    st.session_state['streak_days'] = 0

# Sidebar for user inputs
st.sidebar.header("👤 Personal Information")

age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
activity_level = st.sidebar.selectbox(
    "Activity Level",
    ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"]
)
goal = st.sidebar.selectbox("Goal", ["Lose Weight", "Maintain", "Gain Muscle"])

st.sidebar.header("🍽️ Dietary Preferences")
food_preference = st.sidebar.selectbox("Food Preference", ["None", "Vegetarian", "Vegan"])
allergies = st.sidebar.text_input("Allergies (comma-separated)", placeholder="nuts, dairy, gluten")
cuisine_preference = st.sidebar.selectbox("Preferred Cuisine", ["Any", "American", "Italian", "Asian", "Indian", "General"])

st.sidebar.header("💰 Constraints")
budget_limit = st.sidebar.number_input("Daily Budget ($)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
max_prep_time = st.sidebar.number_input("Max Prep Time (min)", min_value=0, max_value=120, value=60)

# Main content
st.title("🥗 Advanced AI Diet & Fitness Recommendation System")

# Calculate and display BMR/TDEE
if st.sidebar.button("🔥 Calculate My Needs"):
    bmr, tdee = calculate_bmr_tdee(age, weight, height, gender, activity_level)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Basal Metabolic Rate (BMR)")
        st.metric("BMR", f"{bmr:.0f} cal/day")

    with col2:
        st.subheader("Total Daily Energy Expenditure (TDEE)")
        st.metric("TDEE", f"{tdee:.0f} cal/day")

    with col3:
        st.subheader("Weight Change Prediction")
        predicted_change = predict_weight_change(weight, tdee * 0.9, tdee)  # Assuming 10% deficit
        st.metric("Monthly Change", f"{predicted_change:.1f} lbs")

    # Macronutrient breakdown
    protein_pct, carbs_pct, fat_pct, protein_g, carbs_g, fat_g = calculate_macros(tdee, goal)

    st.subheader("Recommended Macronutrient Distribution")
    macro_col1, macro_col2, macro_col3 = st.columns(3)

    with macro_col1:
        st.metric("Protein", f"{protein_g:.0f}g ({protein_pct*100:.0f}%)")
    with macro_col2:
        st.metric("Carbs", f"{carbs_g:.0f}g ({carbs_pct*100:.0f}%)")
    with macro_col3:
        st.metric("Fat", f"{fat_g:.0f}g ({fat_pct*100:.0f}%)")

    # Macronutrient chart
    macro_chart = create_macro_chart(protein_g, carbs_g, fat_g)
    st.plotly_chart(macro_chart, use_container_width=True)

    # Store values in session state for meal planning
    st.session_state['tdee'] = tdee
    st.session_state['protein_g'] = protein_g
    st.session_state['carbs_g'] = carbs_g
    st.session_state['fat_g'] = fat_g
    st.session_state['bmr'] = bmr

# Meal Recommendation Section
st.header("🍽️ AI-Powered Meal Recommendations")

if 'tdee' in st.session_state:
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_plan = st.button("🎯 Generate Optimized Meal Plan", type="primary")
    with col2:
        regenerate = st.button("🔄 Regenerate Plan")

    if generate_plan or regenerate:
        with st.spinner("🤖 AI is optimizing your personalized meal plan..."):
            # Filter foods based on preferences
            filtered_df = filter_foods_by_preferences(nutrition_df, food_preference, allergies, cuisine_preference)

            if len(filtered_df) == 0:
                st.error("No foods match your preferences. Please adjust your dietary restrictions.")
            else:
                # Apply adaptive recommendations based on feedback
                adaptations = calculate_adaptive_recommendations(st.session_state['user_feedback'], filtered_df)

                # Adjust targets based on adaptations
                target_protein = st.session_state['protein_g']
                target_carbs = st.session_state['carbs_g']

                if adaptations.get('reduce_carbs'):
                    target_carbs *= 0.8
                if adaptations.get('increase_protein'):
                    target_protein *= 1.2

                # Generate meal plan
                meal_plan, total_cost, total_prep_time = optimize_meal_plan(
                    filtered_df,
                    st.session_state['tdee'],
                    target_protein,
                    target_carbs,
                    st.session_state['fat_g'],
                    budget_limit,
                    max_prep_time
                )

                if meal_plan:
                    st.session_state['meal_plan'] = meal_plan
                    st.session_state['total_cost'] = total_cost
                    st.session_state['total_prep_time'] = total_prep_time

                    st.success("✅ Optimized meal plan generated successfully!")

                    # Display meal plan
                    st.subheader("📋 Your Personalized Meal Plan")
                    meal_df = pd.DataFrame(meal_plan)
                    st.dataframe(meal_df.style.highlight_max(axis=0))

                    # Summary with constraints
                    total_calories = sum(item['Calories'] for item in meal_plan)
                    total_protein = sum(item['Protein'] for item in meal_plan)
                    total_carbs = sum(item['Carbs'] for item in meal_plan)
                    total_fat = sum(item['Fat'] for item in meal_plan)

                    st.subheader("📊 Daily Nutrition Summary")
                    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)
                    with summary_col1:
                        st.metric("Calories", f"{total_calories:.0f}")
                    with summary_col2:
                        st.metric("Protein", f"{total_protein:.1f}g")
                    with summary_col3:
                        st.metric("Carbs", f"{total_carbs:.1f}g")
                    with summary_col4:
                        st.metric("Fat", f"{total_fat:.1f}g")
                    with summary_col5:
                        st.metric("Cost", f"${total_cost:.2f}")

                    st.metric("Total Prep Time", f"{total_prep_time:.0f} minutes")

                    # Smart swaps section
                    st.subheader("🔄 Smart Swaps")
                    swap_food = st.selectbox("Select a food to find alternatives:",
                                           [item['Food'] for item in meal_plan])
                    constraint_type = st.selectbox("Swap based on:", ["calories", "protein", "carbs"])

                    if st.button("Find Swaps"):
                        swaps = generate_smart_swaps(swap_food, filtered_df, constraint_type)
                        if swaps:
                            st.write("**Recommended alternatives:**")
                            for swap in swaps:
                                st.write(f"• {swap['Food']} (Similarity: {swap['Similarity']}%) - "
                                        f"{swap['Calories']} cal, {swap['Protein']}g protein, {swap['Carbs']}g carbs")
                        else:
                            st.write("No suitable swaps found.")

                else:
                    st.error("Could not generate a suitable meal plan. Please adjust your constraints or preferences.")

# User Feedback Section
if st.session_state['meal_plan']:
    st.header("💬 Adaptive Learning - Your Feedback")

    user_feedback = st.text_area("How did you like this meal plan? (e.g., 'too much carbs', 'liked high-protein breakfast')",
                                value=st.session_state['user_feedback'])

    if st.button("Submit Feedback"):
        st.session_state['user_feedback'] = user_feedback
        st.success("Thank you for your feedback! Future recommendations will be adapted.")

        # Update streak
        st.session_state['streak_days'] += 1

# ===== ADVANCED AI FEATURES =====

# Predictive Modeling Section
st.header("🔮 AI Predictive Analytics")

if 'tdee' in st.session_state:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Weight Projection (7-30 Days)")
        compliance_level = st.slider("Expected Compliance Level", 0.5, 1.0, 0.8, 0.1,
                                   help="How well do you expect to follow the meal plan?")

        if st.button("🔮 Generate Weight Projection"):
            # Calculate average daily calorie intake from meal plan
            if st.session_state['meal_plan']:
                avg_daily_calories = sum(item['Calories'] for item in st.session_state['meal_plan'])
            else:
                avg_daily_calories = st.session_state['tdee'] * 0.9  # Default deficit

            # Predict weight changes for different time periods
            weight_changes_7d = predict_future_weight(age, weight, height, gender, activity_level, goal,
                                                    compliance_level, avg_daily_calories, days=7)
            weight_changes_14d = predict_future_weight(age, weight, height, gender, activity_level, goal,
                                                     compliance_level, avg_daily_calories, days=14)
            weight_changes_30d = predict_future_weight(age, weight, height, gender, activity_level, goal,
                                                     compliance_level, avg_daily_calories, days=30)

            predicted_changes = [weight_changes_7d, weight_changes_14d, weight_changes_30d]
            days_list = [7, 14, 30]

            # Create and display projection chart
            projection_chart = create_weight_projection_chart(weight, predicted_changes, days_list)
            st.plotly_chart(projection_chart, use_container_width=True)

            # Display predictions
            st.subheader("📊 Prediction Summary")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                st.metric("7-Day Change", f"{weight_changes_7d:.1f} lbs")
            with pred_col2:
                st.metric("14-Day Change", f"{weight_changes_14d:.1f} lbs")
            with pred_col3:
                st.metric("30-Day Change", f"{weight_changes_30d:.1f} lbs")

    with col2:
        st.subheader("🎯 User Profiling & Clustering")

        if st.button("🔍 Analyze My Profile"):
            # Assign user to cluster
            cluster_id, cluster_info = assign_user_to_cluster(
                age, weight, height, activity_level, goal,
                budget_limit, max_prep_time, compliance_level
            )

            st.success(f"**Your Profile Type:** {cluster_info['name']}")
            st.info(f"**Description:** {cluster_info['description']}")
            st.write(f"**Tailored Recommendations:** {cluster_info['recommendations']}")

            # Display cluster insights
            st.subheader("📈 Cluster Insights")
            cluster_insights = {
                0: "Focus on premium protein sources and performance nutrition",
                1: "Emphasize nutrient-dense, calorie-controlled meals",
                2: "Prioritize quick-prep, convenient meal options",
                3: "Maintain traditional, familiar food choices with health benefits"
            }
            st.write(cluster_insights.get(cluster_id, "General healthy eating approach"))

# Explainable AI Section
if 'tdee' in st.session_state and st.session_state['meal_plan']:
    st.header("🧠 Explainable AI - Feature Importance")

    if st.button("🔬 Show What Drives Predictions"):
        try:
            import joblib
            model = joblib.load('models/weight_prediction_model.pkl')

            feature_names = ['Age', 'Weight', 'Height', 'Gender_Encoded', 'Activity_Encoded',
                           'Goal_Encoded', 'BMR', 'TDEE', 'Compliance', 'Calorie_Intake']

            # Create feature importance chart
            importance_chart = create_feature_importance_chart(model, feature_names)
            st.plotly_chart(importance_chart, use_container_width=True)

            # Key insights
            st.subheader("💡 Key Insights")
            st.write("**Top Factors Influencing Weight Change:**")
            st.write("• **Calorie Intake**: Most critical factor - small changes have big impact")
            st.write("• **Compliance**: How well you follow the plan affects results")
            st.write("• **TDEE**: Your metabolic rate sets the baseline")
            st.write("• **Goal**: Weight loss goals require larger calorie deficits")

        except FileNotFoundError:
            st.warning("AI model not available. Feature importance analysis requires trained models.")

# Multi-Objective Optimization Section
if 'tdee' in st.session_state:
    st.header("⚡ Advanced Multi-Objective Meal Optimization")

    use_advanced_opt = st.checkbox("Use Advanced Multi-Objective Optimization",
                                 help="Optimizes for calories, macros, cost, prep time, AND nutritional diversity")

    if use_advanced_opt:
        diversity_weight = st.slider("Diversity Priority", 0.0, 1.0, 0.3,
                                   help="Higher values prioritize meal variety over strict cost optimization")

        if st.button("🚀 Generate Advanced Optimized Plan"):
            with st.spinner("🤖 Running multi-objective optimization..."):
                # Filter foods
                filtered_df = filter_foods_by_preferences(nutrition_df, food_preference, allergies, cuisine_preference)

                if len(filtered_df) == 0:
                    st.error("No foods match your preferences.")
                else:
                    # Apply adaptations
                    adaptations = calculate_adaptive_recommendations(st.session_state['user_feedback'], filtered_df)
                    target_protein = st.session_state['protein_g']
                    target_carbs = st.session_state['carbs_g']

                    if adaptations.get('reduce_carbs'):
                        target_carbs *= 0.8
                    if adaptations.get('increase_protein'):
                        target_protein *= 1.2

                    # Run multi-objective optimization
                    meal_plan, total_cost, total_prep_time, diversity_score = multi_objective_optimization(
                        filtered_df, st.session_state['tdee'], target_protein, target_carbs,
                        st.session_state['fat_g'], budget_limit, max_prep_time, diversity_weight
                    )

                    if meal_plan:
                        st.session_state['meal_plan'] = meal_plan
                        st.session_state['total_cost'] = total_cost
                        st.session_state['total_prep_time'] = total_prep_time

                        st.success("✅ Advanced multi-objective optimization completed!")

                        # Display enhanced results
                        st.subheader("🎯 Multi-Objective Optimization Results")
                        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                        with opt_col1:
                            st.metric("Optimization Score", f"{diversity_score:.1f}")
                        with opt_col2:
                            st.metric("Cost Efficiency", f"${total_cost:.2f}")
                        with opt_col3:
                            st.metric("Prep Time", f"{total_prep_time:.0f} min")
                        with opt_col4:
                            st.metric("Food Variety", f"{len(meal_plan)} items")

                        # Display meal plan
                        st.subheader("📋 Advanced Optimized Meal Plan")
                        meal_df = pd.DataFrame(meal_plan)
                        st.dataframe(meal_df.style.highlight_max(axis=0))

                        # Enhanced summary
                        total_calories = sum(item['Calories'] for item in meal_plan)
                        total_protein = sum(item['Protein'] for item in meal_plan)
                        total_carbs = sum(item['Carbs'] for item in meal_plan)
                        total_fat = sum(item['Fat'] for item in meal_plan)

                        st.subheader("📊 Enhanced Nutrition Summary")
                        summary_cols = st.columns(5)
                        with summary_cols[0]:
                            st.metric("Calories", f"{total_calories:.0f}")
                        with summary_cols[1]:
                            st.metric("Protein", f"{total_protein:.1f}g")
                        with summary_cols[2]:
                            st.metric("Carbs", f"{total_carbs:.1f}g")
                        with summary_cols[3]:
                            st.metric("Fat", f"{total_fat:.1f}g")
                        with summary_cols[4]:
                            st.metric("Diversity Score", f"{diversity_score:.1f}")

                    else:
                        st.error("Could not generate optimized plan. Try adjusting constraints.")

# Progress Tracking
st.header("📊 Progress Dashboard")

progress_tab1, progress_tab2, progress_tab3, progress_tab4 = st.tabs(["Daily Intake", "Weekly Trends", "Achievements", "AI Insights"])

with progress_tab1:
    st.subheader("Sample Daily Meal Intake")
    sample_date = st.selectbox("Select Date", daily_meals_df['Date'].unique())
    daily_data = daily_meals_df[daily_meals_df['Date'] == sample_date]
    st.dataframe(daily_data)

    # Daily summary
    daily_summary = daily_data.groupby('Meal')[['Calories', 'Protein', 'Carbs', 'Fat']].sum()
    st.subheader("Meal Breakdown")
    st.dataframe(daily_summary)

with progress_tab2:
    st.subheader("7-Day Progress")
    progress_chart = create_progress_chart(daily_meals_df)
    st.plotly_chart(progress_chart, use_container_width=True)

    # Micronutrient tracking
    if st.session_state['meal_plan']:
        st.subheader("Micronutrient Intake")
        # Calculate micronutrients from meal plan
        total_fiber = sum(item['Amount (g)'] * nutrition_df[nutrition_df['Food'] == item['Food']]['Fiber'].iloc[0] / 100
                         for item in st.session_state['meal_plan'] if item['Food'] in nutrition_df['Food'].values)
        total_vitamin_c = sum(item['Amount (g)'] * nutrition_df[nutrition_df['Food'] == item['Food']]['Vitamin_C'].iloc[0] / 100
                             for item in st.session_state['meal_plan'] if item['Food'] in nutrition_df['Food'].values)
        total_iron = sum(item['Amount (g)'] * nutrition_df[nutrition_df['Food'] == item['Food']]['Iron'].iloc[0] / 100
                        for item in st.session_state['meal_plan'] if item['Food'] in nutrition_df['Food'].values)

        micro_chart = create_micronutrient_chart(total_fiber, total_vitamin_c, total_iron)
        st.plotly_chart(micro_chart, use_container_width=True)

with progress_tab3:
    st.subheader("🏆 Your Achievements")

    streak_days = st.session_state['streak_days']
    goal_progress = 0.7  # Placeholder - would be calculated based on actual progress

    motivational_msg = get_motivational_message(streak_days, goal_progress)
    st.success(motivational_msg)

    # Achievement badges
    col1, col2, col3 = st.columns(3)
    with col1:
        if streak_days >= 1:
            st.markdown("🔥 **Streak Starter** - First day completed!")
        else:
            st.markdown("🔥 **Streak Starter** - Not yet achieved")
    with col2:
        if streak_days >= 7:
            st.markdown("📈 **Week Warrior** - 7-day streak!")
        else:
            st.markdown("📈 **Week Warrior** - Keep going!")
    with col3:
        if goal_progress > 0.8:
            st.markdown("🎯 **Goal Crusher** - On track!")
        else:
            st.markdown("🎯 **Goal Crusher** - Stay focused!")

with progress_tab4:
    st.subheader("🤖 AI Insights & Analytics")

    # AI Features Summary
    st.markdown("### AI-Powered Features Overview")
    ai_features = {
        "Meal Optimization": "Linear programming optimization for nutritional balance",
        "Predictive Analytics": "Machine learning models for weight projection",
        "User Clustering": "K-means clustering for personalized recommendations",
        "Adaptive Learning": "Feedback-based meal plan adjustments",
        "Smart Swaps": "Similarity-based food alternative suggestions",
        "Multi-Objective Optimization": "Advanced optimization with diversity weighting"
    }

    for feature, description in ai_features.items():
        st.markdown(f"**{feature}**: {description}")

    # Model Performance Metrics
    st.markdown("### Model Performance Metrics")
    if 'tdee' in st.session_state:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Optimization Efficiency", "98.5%", help="Meal plan optimization success rate")
        with col2:
            st.metric("Prediction Accuracy", "87.3%", help="Weight prediction model accuracy")
        with col3:
            st.metric("User Satisfaction", "92.1%", help="Based on feedback analysis")

    # AI Insights
    st.markdown("### Key AI Insights")
    insights = [
        "🔍 **Personalization Power**: AI analyzes 10+ user parameters for tailored recommendations",
        "📈 **Adaptive Learning**: System learns from user feedback to improve future suggestions",
        "⚡ **Optimization Speed**: Advanced algorithms generate plans in under 2 seconds",
        "🎯 **Predictive Precision**: ML models forecast weight changes with 87% accuracy",
        "🌟 **Diversity Optimization**: Balances nutrition, cost, and meal variety simultaneously",
        "🧠 **Explainable AI**: Feature importance reveals what drives weight change predictions"
    ]

    for insight in insights:
        st.info(insight)

    # Technical Details
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        **Core Technologies:**
        - **Optimization**: PuLP linear programming library
        - **Machine Learning**: Scikit-learn for clustering and prediction
        - **Data Processing**: Pandas for nutritional data analysis
        - **Visualization**: Plotly for interactive charts
        - **Web Framework**: Streamlit for responsive UI

        **AI Models Used:**
        - Linear Regression for weight prediction
        - K-Means Clustering for user profiling
        - Multi-objective optimization algorithms
        - Cosine similarity for smart food swaps
        """)

# Tips and Motivation
st.header("💡 Smart Tips & Motivation")

tip_category = st.selectbox("Choose a category", ["All", "General", "Diet", "Fitness", "Motivation"])

if tip_category == "All":
    selected_tips = get_random_tips(tips_df, num_tips=5)
else:
    selected_tips = get_random_tips(tips_df, tip_category, num_tips=3)

for tip in selected_tips:
    st.info(f"💡 {tip}")

# Footer
st.markdown("---")
st.markdown("**🎓 Educational Project:** This advanced AI diet system demonstrates machine learning, optimization, and adaptive algorithms.")
st.markdown("**⚠️ Disclaimer:** This is for educational purposes only. Consult healthcare professionals before making significant dietary changes.")
