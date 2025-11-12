import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def calculate_bmr_tdee(age, weight, height, gender, activity_level):
    """
    Calculate BMR using Mifflin-St Jeor equation and TDEE based on activity level.
    """
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multipliers = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }

    tdee = bmr * activity_multipliers.get(activity_level.lower(), 1.2)
    return bmr, tdee

def calculate_macros(calories, goal):
    """
    Calculate macronutrient ratios based on goal.
    Returns protein, carbs, fat percentages and grams.
    """
    if goal.lower() == 'lose weight':
        protein_pct, carbs_pct, fat_pct = 0.35, 0.40, 0.25
    elif goal.lower() == 'gain muscle':
        protein_pct, carbs_pct, fat_pct = 0.30, 0.50, 0.20
    else:  # maintain
        protein_pct, carbs_pct, fat_pct = 0.25, 0.50, 0.25

    protein_g = (calories * protein_pct) / 4
    carbs_g = (calories * carbs_pct) / 4
    fat_g = (calories * fat_pct) / 9

    return protein_pct, carbs_pct, fat_pct, protein_g, carbs_g, fat_g

def filter_foods_by_preferences(nutrition_df, food_preference, allergies, cuisine_preference):
    """
    Filter foods based on user preferences, allergies, and cuisine.
    """
    filtered_df = nutrition_df.copy()

    # Filter by food preference
    if food_preference.lower() == 'vegetarian':
        non_veg_foods = ['Chicken Breast', 'Salmon', 'Beef', 'Tuna']
        filtered_df = filtered_df[~filtered_df['Food'].isin(non_veg_foods)]
    elif food_preference.lower() == 'vegan':
        non_vegan_foods = ['Chicken Breast', 'Salmon', 'Beef', 'Tuna', 'Egg', 'Milk', 'Greek Yogurt',
                           'Cheese', 'Butter', 'Yogurt', 'Cottage Cheese']
        filtered_df = filtered_df[~filtered_df['Food'].isin(non_vegan_foods)]

    # Filter by allergies
    if allergies:
        allergy_list = [allergy.strip().lower() for allergy in allergies.split(',')]
        for allergy in allergy_list:
            if allergy in ['nuts', 'peanuts']:
                filtered_df = filtered_df[~filtered_df['Allergens'].str.lower().str.contains('nuts')]
            elif allergy == 'dairy':
                filtered_df = filtered_df[~filtered_df['Allergens'].str.lower().str.contains('dairy')]
            elif allergy == 'fish':
                filtered_df = filtered_df[~filtered_df['Allergens'].str.lower().str.contains('fish')]
            elif allergy == 'soy':
                filtered_df = filtered_df[~filtered_df['Allergens'].str.lower().str.contains('soy')]
            elif allergy == 'gluten':
                filtered_df = filtered_df[~filtered_df['Allergens'].str.lower().str.contains('gluten')]

    # Filter by cuisine preference
    if cuisine_preference and cuisine_preference.lower() != 'any':
        filtered_df = filtered_df[filtered_df['Cuisine'].str.lower() == cuisine_preference.lower()]

    return filtered_df

def optimize_meal_plan(nutrition_df, target_calories, target_protein, target_carbs, target_fat,
                      budget_limit=None, max_prep_time=None, num_foods=10):
    """
    Use linear programming to optimize meal selection with constraints.
    """
    # Create the LP problem
    prob = LpProblem("Meal_Optimization", LpMinimize)

    # Decision variables: amount of each food (0 to 200g)
    food_vars = LpVariable.dicts("Food", nutrition_df.index, lowBound=0, upBound=200)

    # Objective: minimize total cost (if budget limit provided) or calories
    if budget_limit:
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Cost'] / 100) for i in nutrition_df.index])
    else:
        prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index])

    # Basic nutritional constraints
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index]) >= target_calories * 0.9
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index]) <= target_calories * 1.1

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Protein'] for i in nutrition_df.index]) >= target_protein * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Protein'] for i in nutrition_df.index]) <= target_protein * 1.2

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Carbs'] for i in nutrition_df.index]) >= target_carbs * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Carbs'] for i in nutrition_df.index]) <= target_carbs * 1.2

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Fat'] for i in nutrition_df.index]) >= target_fat * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Fat'] for i in nutrition_df.index]) <= target_fat * 1.2

    # Budget constraint
    if budget_limit:
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Cost'] / 100) for i in nutrition_df.index]) <= budget_limit

    # Prep time constraint
    if max_prep_time:
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Prep_Time'] / 100) for i in nutrition_df.index]) <= max_prep_time

    # Solve the problem
    status = prob.solve()

    # Extract results
    selected_foods = []
    total_cost = 0
    total_prep_time = 0

    for i in nutrition_df.index:
        amount = food_vars[i].varValue
        if amount > 0.1:  # Only include foods with meaningful amounts
            food_info = nutrition_df.loc[i]
            cost = amount * food_info['Cost'] / 100
            prep_time = amount * food_info['Prep_Time'] / 100
            total_cost += cost
            total_prep_time += prep_time

            selected_foods.append({
                'Food': food_info['Food'],
                'Amount (g)': round(amount, 1),
                'Calories': round(amount * food_info['Calories'] / 100, 1),
                'Protein': round(amount * food_info['Protein'] / 100, 1),
                'Carbs': round(amount * food_info['Carbs'] / 100, 1),
                'Fat': round(amount * food_info['Fat'] / 100, 1),
                'Cost ($)': round(cost, 2),
                'Prep Time (min)': round(prep_time, 1),
                'Cuisine': food_info['Cuisine']
            })

    return selected_foods[:num_foods], round(total_cost, 2), round(total_prep_time, 1)

def generate_smart_swaps(food_name, nutrition_df, constraint_type):
    """
    Generate smart swap suggestions based on nutritional profile.
    """
    if food_name not in nutrition_df['Food'].values:
        return []

    food_data = nutrition_df[nutrition_df['Food'] == food_name].iloc[0]
    swaps = []

    # Find foods with similar nutritional profiles
    for idx, row in nutrition_df.iterrows():
        if row['Food'] != food_name:
            similarity_score = 0

            # Compare based on constraint type
            if constraint_type == 'calories':
                cal_diff = abs(row['Calories'] - food_data['Calories'])
                similarity_score = max(0, 100 - cal_diff)
            elif constraint_type == 'protein':
                prot_diff = abs(row['Protein'] - food_data['Protein'])
                similarity_score = max(0, 20 - prot_diff)
            elif constraint_type == 'carbs':
                carb_diff = abs(row['Carbs'] - food_data['Carbs'])
                similarity_score = max(0, 50 - carb_diff)

            if similarity_score > 50:  # Good similarity
                swaps.append({
                    'Food': row['Food'],
                    'Similarity': round(similarity_score, 1),
                    'Calories': row['Calories'],
                    'Protein': row['Protein'],
                    'Carbs': row['Carbs']
                })

    return sorted(swaps, key=lambda x: x['Similarity'], reverse=True)[:3]

def create_macro_chart(protein_g, carbs_g, fat_g):
    """
    Create a pie chart for macronutrient distribution.
    """
    labels = ['Protein', 'Carbs', 'Fat']
    values = [protein_g, carbs_g, fat_g]
    colors = ['#FF9999', '#66B3FF', '#99FF99']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors)])
    fig.update_layout(title_text="Macronutrient Distribution")
    return fig

def create_micronutrient_chart(fiber_g, vitamin_c_mg, iron_mg):
    """
    Create a bar chart for micronutrient intake.
    """
    nutrients = ['Fiber (g)', 'Vitamin C (mg)', 'Iron (mg)']
    values = [fiber_g, vitamin_c_mg, iron_mg]
    colors = ['#8B4513', '#FF6347', '#DC143C']

    fig = go.Figure(data=[go.Bar(x=nutrients, y=values, marker_color=colors)])
    fig.update_layout(title_text="Micronutrient Intake", xaxis_title="Nutrient", yaxis_title="Amount")
    return fig

def create_progress_chart(daily_meals_df):
    """
    Create a line chart showing daily calorie intake over time.
    """
    daily_calories = daily_meals_df.groupby('Date')['Calories'].sum().reset_index()

    fig = px.line(daily_calories, x='Date', y='Calories', title='Daily Calorie Intake')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Calories')
    return fig

def predict_weight_change(current_weight, daily_calorie_intake, tdee, days=30):
    """
    Predict potential weight change based on calorie surplus/deficit.
    """
    daily_deficit = tdee - daily_calorie_intake
    weekly_change = (daily_deficit * 7) / 3500  # 1 lb = 3500 calories
    monthly_change = weekly_change * 4.3  # Approximate month

    return round(monthly_change, 1)

def get_random_tips(tips_df, category=None, num_tips=3):
    """
    Get random tips from the tips dataset.
    """
    if category:
        filtered_tips = tips_df[tips_df['Category'] == category]
    else:
        filtered_tips = tips_df

    return filtered_tips.sample(min(num_tips, len(filtered_tips)))['Tip'].tolist()

def calculate_adaptive_recommendations(user_feedback, nutrition_df):
    """
    Adapt recommendations based on user feedback.
    """
    adaptations = {}

    if user_feedback:
        feedback_lower = user_feedback.lower()

        if 'too much carbs' in feedback_lower:
            adaptations['reduce_carbs'] = True
        if 'high protein' in feedback_lower or 'more protein' in feedback_lower:
            adaptations['increase_protein'] = True
        if 'spicy' in feedback_lower:
            adaptations['reduce_spicy'] = True
        if 'bland' in feedback_lower:
            adaptations['increase_flavor'] = True

    return adaptations

def get_motivational_message(streak_days, goal_progress):
    """
    Generate motivational messages based on user progress.
    """
    messages = {
        'streak': [
            f"🔥 {streak_days} day streak! You're on fire!",
            f"📈 {streak_days} consecutive days of healthy eating!",
            f"💪 Consistency is key, and you're crushing it with {streak_days} days!"
        ],
        'progress': [
            "🎯 You're making great progress toward your goals!",
            "🌟 Small changes add up to big results!",
            "🚀 You're on the path to success!"
        ],
        'encouragement': [
            "💪 Every healthy choice counts!",
            "🌱 Growth happens one meal at a time!",
            "🎉 Celebrate your commitment to health!"
        ]
    }

    if streak_days > 0:
        return random.choice(messages['streak'])
    elif goal_progress > 0.8:
        return random.choice(messages['progress'])
    else:
        return random.choice(messages['encouragement'])

# ===== NEW ADVANCED ML FEATURES =====

def train_weight_prediction_model():
    """
    Train a Random Forest model to predict future weight based on user profile and compliance.
    """
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic user profiles
    ages = np.random.randint(18, 65, n_samples)
    weights = np.random.normal(70, 15, n_samples)
    heights = np.random.normal(170, 10, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples)
    activity_levels = np.random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'], n_samples)
    goals = np.random.choice(['Lose Weight', 'Maintain', 'Gain Muscle'], n_samples)

    # Calculate BMR and TDEE
    bmr_values = []
    tdee_values = []
    for i in range(n_samples):
        if genders[i] == 'Male':
            bmr = 10 * weights[i] + 6.25 * heights[i] - 5 * ages[i] + 5
        else:
            bmr = 10 * weights[i] + 6.25 * heights[i] - 5 * ages[i] - 161

        activity_multipliers = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725
        }
        tdee = bmr * activity_multipliers[activity_levels[i]]
        bmr_values.append(bmr)
        tdee_values.append(tdee)

    # Compliance factors (0-1 scale)
    compliance = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic compliance

    # Calculate calorie intake (with some variation based on compliance)
    calorie_intake = []
    for i in range(n_samples):
        if goals[i] == 'Lose Weight':
            target = tdee_values[i] * 0.8  # 20% deficit
        elif goals[i] == 'Gain Muscle':
            target = tdee_values[i] * 1.2  # 20% surplus
        else:
            target = tdee_values[i]  # Maintenance

        # Actual intake varies based on compliance
        actual_intake = target * (0.8 + compliance[i] * 0.4)  # 80-120% of target
        calorie_intake.append(actual_intake)

    # Calculate weight change after 30 days
    weight_changes = []
    for i in range(n_samples):
        daily_deficit = tdee_values[i] - calorie_intake[i]
        weekly_change = (daily_deficit * 7) / 3500  # 1 lb = 3500 calories
        monthly_change = weekly_change * 4.3  # Approximate month
        weight_changes.append(monthly_change)

    # Create DataFrame
    data = pd.DataFrame({
        'Age': ages,
        'Weight': weights,
        'Height': heights,
        'Gender': genders,
        'Activity_Level': activity_levels,
        'Goal': goals,
        'BMR': bmr_values,
        'TDEE': tdee_values,
        'Compliance': compliance,
        'Calorie_Intake': calorie_intake,
        'Weight_Change_30d': weight_changes
    })

    # Encode categorical variables
    le_gender = LabelEncoder()
    le_activity = LabelEncoder()
    le_goal = LabelEncoder()

    data['Gender_Encoded'] = le_gender.fit_transform(data['Gender'])
    data['Activity_Encoded'] = le_activity.fit_transform(data['Activity_Level'])
    data['Goal_Encoded'] = le_goal.fit_transform(data['Goal'])

    # Features and target
    features = ['Age', 'Weight', 'Height', 'Gender_Encoded', 'Activity_Encoded', 'Goal_Encoded',
               'BMR', 'TDEE', 'Compliance', 'Calorie_Intake']
    X = data[features]
    y = data['Weight_Change_30d']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Model RMSE: {rmse:.2f} lbs")

    # Save model and encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/weight_prediction_model.pkl')
    joblib.dump(le_gender, 'models/gender_encoder.pkl')
    joblib.dump(le_activity, 'models/activity_encoder.pkl')
    joblib.dump(le_goal, 'models/goal_encoder.pkl')

    return model, rmse

def predict_future_weight(age, weight, height, gender, activity_level, goal, compliance, calorie_intake, days=30):
    """
    Predict future weight change using trained model.
    """
    try:
        # Load model and encoders
        model = joblib.load('models/weight_prediction_model.pkl')
        le_gender = joblib.load('models/gender_encoder.pkl')
        le_activity = joblib.load('models/activity_encoder.pkl')
        le_goal = joblib.load('models/goal_encoder.pkl')

        # Calculate BMR and TDEE
        if gender == 'Male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_multipliers = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725,
            'Extra Active': 1.9
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.2)

        # Encode categorical variables
        gender_encoded = le_gender.transform([gender])[0]
        activity_encoded = le_activity.transform([activity_level])[0]
        goal_encoded = le_goal.transform([goal])[0]

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Weight': [weight],
            'Height': [height],
            'Gender_Encoded': [gender_encoded],
            'Activity_Encoded': [activity_encoded],
            'Goal_Encoded': [goal_encoded],
            'BMR': [bmr],
            'TDEE': [tdee],
            'Compliance': [compliance],
            'Calorie_Intake': [calorie_intake]
        })

        # Predict weight change
        weight_change = model.predict(input_data)[0]

        # Scale for different time periods
        if days == 7:
            predicted_change = weight_change * (7/30)
        elif days == 14:
            predicted_change = weight_change * (14/30)
        elif days == 30:
            predicted_change = weight_change
        else:
            predicted_change = weight_change * (days/30)

        return round(predicted_change, 1)

    except FileNotFoundError:
        # Fallback to simple calculation if model not available
        if gender == 'Male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_multipliers = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725,
            'Extra Active': 1.9
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.2)

        daily_deficit = tdee - calorie_intake
        weekly_change = (daily_deficit * 7) / 3500
        predicted_change = weekly_change * (days/7)

        return round(predicted_change, 1)

def create_weight_projection_chart(current_weight, predicted_changes, days_list):
    """
    Create a chart showing weight projection over time.
    """
    dates = [datetime.now() + timedelta(days=d) for d in days_list]
    weights = [current_weight + change for change in predicted_changes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=weights, mode='lines+markers',
                            name='Predicted Weight', line=dict(color='#FF6B6B', width=3)))

    fig.add_trace(go.Scatter(x=[datetime.now()], y=[current_weight], mode='markers',
                            name='Current Weight', marker=dict(size=10, color='#4ECDC4')))

    fig.update_layout(
        title="30-Day Weight Projection",
        xaxis_title="Date",
        yaxis_title="Weight (lbs)",
        showlegend=True
    )

    return fig

def perform_user_clustering():
    """
    Perform K-Means clustering on user profiles to create user segments.
    """
    # Create synthetic user data for clustering
    np.random.seed(42)
    n_users = 500

    users = pd.DataFrame({
        'Age': np.random.randint(18, 65, n_users),
        'Weight': np.random.normal(70, 15, n_users),
        'Height': np.random.normal(170, 10, n_users),
        'Activity_Level': np.random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'], n_users),
        'Goal': np.random.choice(['Lose Weight', 'Maintain', 'Gain Muscle'], n_users),
        'Budget': np.random.uniform(5, 25, n_users),
        'Prep_Time_Pref': np.random.uniform(15, 90, n_users),
        'Compliance_Score': np.random.beta(2, 2, n_users)
    })

    # Encode categorical variables
    le_activity = LabelEncoder()
    le_goal = LabelEncoder()

    users['Activity_Encoded'] = le_activity.fit_transform(users['Activity_Level'])
    users['Goal_Encoded'] = le_goal.fit_transform(users['Goal'])

    # Features for clustering
    features = ['Age', 'Weight', 'Height', 'Activity_Encoded', 'Goal_Encoded',
               'Budget', 'Prep_Time_Pref', 'Compliance_Score']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(users[features])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    users['Cluster'] = clusters

    # Define cluster characteristics
    cluster_profiles = {
        0: {
            'name': 'Fitness Enthusiasts',
            'description': 'Young, active individuals focused on muscle gain with higher budgets',
            'recommendations': 'High-protein, performance-focused meals with premium ingredients'
        },
        1: {
            'name': 'Weight Loss Warriors',
            'description': 'Motivated individuals with moderate activity levels seeking weight loss',
            'recommendations': 'Calorie-controlled, nutrient-dense meals with cost optimization'
        },
        2: {
            'name': 'Busy Professionals',
            'description': 'Time-constrained individuals with moderate budgets and prep time limits',
            'recommendations': 'Quick-prep, balanced meals with convenience focus'
        },
        3: {
            'name': 'Health Conscious Traditionalists',
            'description': 'Older individuals maintaining health with conservative approaches',
            'recommendations': 'Traditional, familiar foods with nutritional balance'
        }
    }

    return users, kmeans, scaler, cluster_profiles

def assign_user_to_cluster(age, weight, height, activity_level, goal, budget, prep_time_pref, compliance_score=0.7):
    """
    Assign a new user to a cluster based on their profile.
    """
    try:
        # Load clustering model
        kmeans = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/cluster_scaler.pkl')
        le_activity = joblib.load('models/activity_encoder.pkl')
        le_goal = joblib.load('models/goal_encoder.pkl')
        cluster_profiles = joblib.load('models/cluster_profiles.pkl')

        # Encode categorical variables
        activity_encoded = le_activity.transform([activity_level])[0]
        goal_encoded = le_goal.transform([goal])[0]

        # Create user data
        user_data = pd.DataFrame({
            'Age': [age],
            'Weight': [weight],
            'Height': [height],
            'Activity_Encoded': [activity_encoded],
            'Goal_Encoded': [goal_encoded],
            'Budget': [budget],
            'Prep_Time_Pref': [prep_time_pref],
            'Compliance_Score': [compliance_score]
        })

        # Scale features
        user_scaled = scaler.transform(user_data)

        # Predict cluster
        cluster = kmeans.predict(user_scaled)[0]

        return cluster, cluster_profiles[cluster]

    except FileNotFoundError:
        # Default clustering based on simple rules
        if age < 30 and goal == 'Gain Muscle':
            cluster = 0  # Fitness Enthusiasts
        elif goal == 'Lose Weight':
            cluster = 1  # Weight Loss Warriors
        elif prep_time_pref < 30:
            cluster = 2  # Busy Professionals
        else:
            cluster = 3  # Health Conscious Traditionalists

        cluster_profiles = {
            0: {
                'name': 'Fitness Enthusiasts',
                'description': 'Young, active individuals focused on muscle gain',
                'recommendations': 'High-protein, performance-focused meals'
            },
            1: {
                'name': 'Weight Loss Warriors',
                'description': 'Individuals seeking weight loss',
                'recommendations': 'Calorie-controlled, nutrient-dense meals'
            },
            2: {
                'name': 'Busy Professionals',
                'description': 'Time-constrained individuals',
                'recommendations': 'Quick-prep, balanced meals'
            },
            3: {
                'name': 'Health Conscious Traditionalists',
                'description': 'Individuals maintaining health with conservative approaches',
                'recommendations': 'Traditional, familiar foods with nutritional balance'
            }
        }

        return cluster, cluster_profiles[cluster]

def create_feature_importance_chart(model, feature_names):
    """
    Create a chart showing feature importance for explainable AI.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        marker_color='#4ECDC4'
    ))

    fig.update_layout(
        title="Feature Importance for Weight Prediction",
        xaxis_title="Features",
        yaxis_title="Importance Score",
        xaxis_tickangle=-45
    )

    return fig

def multi_objective_optimization(nutrition_df, target_calories, target_protein, target_carbs, target_fat,
                               budget_limit=None, max_prep_time=None, diversity_weight=0.1):
    """
    Advanced multi-objective optimization for meal planning.
    Optimizes for calories, macros, cost, prep time, and nutritional diversity.
    """
    # Create the LP problem with multiple objectives
    prob = LpProblem("Multi_Objective_Meal_Optimization", LpMinimize)

    # Decision variables: amount of each food (0 to 200g)
    food_vars = LpVariable.dicts("Food", nutrition_df.index, lowBound=0, upBound=200)

    # Multi-objective function: minimize cost + diversity penalty
    if budget_limit:
        # Weighted objective: cost + diversity penalty
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Cost'] / 100) for i in nutrition_df.index]) + \
               diversity_weight * lpSum([food_vars[i] for i in nutrition_df.index])  # Diversity penalty
    else:
        prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index])

    # Nutritional constraints
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index]) >= target_calories * 0.9
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Calories'] for i in nutrition_df.index]) <= target_calories * 1.1

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Protein'] for i in nutrition_df.index]) >= target_protein * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Protein'] for i in nutrition_df.index]) <= target_protein * 1.2

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Carbs'] for i in nutrition_df.index]) >= target_carbs * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Carbs'] for i in nutrition_df.index]) <= target_carbs * 1.2

    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Fat'] for i in nutrition_df.index]) >= target_fat * 0.8
    prob += lpSum([food_vars[i] * nutrition_df.loc[i, 'Fat'] for i in nutrition_df.index]) <= target_fat * 1.2

    # Additional constraints
    if budget_limit:
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Cost'] / 100) for i in nutrition_df.index]) <= budget_limit

    if max_prep_time:
        prob += lpSum([food_vars[i] * (nutrition_df.loc[i, 'Prep_Time'] / 100) for i in nutrition_df.index]) <= max_prep_time

    # Diversity constraint: limit maximum amount per food to encourage variety
    for i in nutrition_df.index:
        prob += food_vars[i] <= 150  # Max 150g per food

    # Solve the problem
    status = prob.solve()

    # Extract results
    selected_foods = []
    total_cost = 0
    total_prep_time = 0
    diversity_score = 0

    for i in nutrition_df.index:
        amount = food_vars[i].varValue
        if amount > 0.1:
            food_info = nutrition_df.loc[i]
            cost = amount * food_info['Cost'] / 100
            prep_time = amount * food_info['Prep_Time'] / 100
            total_cost += cost
            total_prep_time += prep_time
            diversity_score += 1  # Count of different foods

            selected_foods.append({
                'Food': food_info['Food'],
                'Amount (g)': round(amount, 1),
                'Calories': round(amount * food_info['Calories'] / 100, 1),
                'Protein': round(amount * food_info['Protein'] / 100, 1),
                'Carbs': round(amount * food_info['Carbs'] / 100, 1),
                'Fat': round(amount * food_info['Fat'] / 100, 1),
                'Cost ($)': round(cost, 2),
                'Prep Time (min)': round(prep_time, 1),
                'Cuisine': food_info['Cuisine']
            })

    return selected_foods[:10], round(total_cost, 2), round(total_prep_time, 1), diversity_score
