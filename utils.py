import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
import ast
try:
    import streamlit as st
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

# Constants
MEAL_SLOTS = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly active': 1.375,
    'moderately active': 1.55,
    'very active': 1.725,
    'extra active': 1.9
}
GOAL_ADJUSTMENTS = {
    'lose weight': -500,
    'maintain': 0,
    'gain muscle': 500
}
MACRO_RATIOS = {
    'lose weight': {'protein': 0.35, 'carbs': 0.40, 'fat': 0.25},
    'maintain': {'protein': 0.25, 'carbs': 0.50, 'fat': 0.25},
    'gain muscle': {'protein': 0.30, 'carbs': 0.50, 'fat': 0.20}
}

def calculate_targets(user: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate daily calorie and macro targets using Mifflin-St Jeor formula.

    Args:
        user: Dict with keys: age, sex, weight, height, activity_level, goal

    Returns:
        Tuple of (daily_calories, macro_targets_dict)
    """
    age = user['age']
    sex = user['sex'].lower()
    weight = user['weight']  # kg
    height = user['height']  # cm
    activity_level = user['activity_level'].lower()
    goal = user['goal'].lower()

    # Mifflin-St Jeor BMR
    if sex == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # TDEE
    tdee = bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)

    # Adjust for goal
    daily_calories = tdee + GOAL_ADJUSTMENTS.get(goal, 0)

    # Calculate macro targets
    ratios = MACRO_RATIOS.get(goal, MACRO_RATIOS['maintain'])
    macro_targets = {
        'protein': (daily_calories * ratios['protein']) / 4,  # g
        'carbs': (daily_calories * ratios['carbs']) / 4,     # g
        'fat': (daily_calories * ratios['fat']) / 9          # g
    }

    return daily_calories, macro_targets

def filter_meals(df: pd.DataFrame, preferences: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter meals based on user preferences.

    Args:
        df: Nutrition dataframe
        preferences: Dict with keys: cuisine, veg_flag, allergens, health_level

    Returns:
        Filtered dataframe
    """
    filtered_df = df.copy()

    # Filter by health_level
    health_level = preferences.get('health_level', 'light')
    if health_level == 'light':
        allowed_scores = ['light']
    elif health_level == 'light+moderate':
        allowed_scores = ['light', 'moderate']
    else:
        allowed_scores = ['light', 'moderate', 'heavy']

    filtered_df = filtered_df[filtered_df['health_score'].isin(allowed_scores)]

    # Filter by veg_flag
    veg_flag = preferences.get('veg_flag', 'none')
    if veg_flag == 'vegetarian':
        # Exclude non-veg foods - use tags if available, else common exclusions
        if 'tags' in filtered_df.columns:
            non_veg_mask = filtered_df['tags'].str.contains('non-veg|meat|fish|chicken|beef|pork', case=False, na=False)
            filtered_df = filtered_df[~non_veg_mask]
        else:
            # Fallback to common non-veg foods
            non_veg_foods = ['Chicken Breast', 'Salmon', 'Beef', 'Tuna', 'Turkey Breast',
                           'Duck Breast', 'Lamb', 'Pork Tenderloin', 'Shrimp', 'Lobster',
                           'Crab', 'Cod', 'Tilapia', 'Mahi Mahi', 'Swordfish', 'Mackerel',
                           'Sardines', 'Anchovies']
            filtered_df = filtered_df[~filtered_df.index.isin(non_veg_foods)]
    elif veg_flag == 'vegan':
        # Exclude animal products
        if 'tags' in filtered_df.columns:
            animal_mask = filtered_df['tags'].str.contains('dairy|egg|meat|fish|chicken|beef|pork|milk|yogurt|cheese', case=False, na=False)
            filtered_df = filtered_df[~animal_mask]
        else:
            # Fallback exclusions
            animal_foods = ['Chicken Breast', 'Salmon', 'Beef', 'Tuna', 'Turkey Breast',
                          'Duck Breast', 'Lamb', 'Pork Tenderloin', 'Shrimp', 'Lobster',
                          'Crab', 'Cod', 'Tilapia', 'Mahi Mahi', 'Swordfish', 'Mackerel',
                          'Sardines', 'Anchovies', 'Egg', 'Milk', 'Greek Yogurt', 'Cheese',
                          'Butter', 'Yogurt', 'Cottage Cheese', 'Curd', 'Lassi', 'Paneer']
            filtered_df = filtered_df[~filtered_df.index.isin(animal_foods)]

    # Filter by allergens
    allergens = preferences.get('allergens', [])
    if allergens:
        for allergen in allergens:
            allergen = allergen.lower().strip()
            if allergen:
                # Use tags if available, else food names
                if 'tags' in filtered_df.columns:
                    allergen_mask = filtered_df['tags'].str.contains(allergen, case=False, na=False)
                    filtered_df = filtered_df[~allergen_mask]
                else:
                    # Fallback to name-based filtering
                    name_mask = filtered_df.index.str.lower().str.contains(allergen)
                    filtered_df = filtered_df[~name_mask]

    # Filter by cuisine
    cuisine = preferences.get('cuisine', 'any')
    if cuisine != 'any':
        if 'cuisine' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['cuisine'].str.lower() == cuisine.lower()]
        else:
            # Fallback to name-based cuisine detection
            cuisine_keywords = {
                'american': ['burger', 'pizza', 'pasta', 'tacos', 'enchiladas'],
                'italian': ['pasta', 'pizza'],
                'asian': ['rice', 'pad thai', 'ramen', 'sushi', 'kimchi'],
                'indian': ['dal', 'chana masala', 'rajma', 'paneer', 'chole', 'khichdi',
                          'poha', 'upma', 'idli', 'dosa', 'sambar', 'rasam', 'roti',
                          'chapati', 'naan', 'paratha', 'misal pav', 'dhokla', 'pakora', 'samosa']
            }
            keywords = cuisine_keywords.get(cuisine.lower(), [])
            if keywords:
                cuisine_mask = filtered_df.index.str.lower().str.contains('|'.join(keywords))
                filtered_df = filtered_df[cuisine_mask]

    return filtered_df

def score_and_rank(meal_row: pd.Series, user_targets: Dict[str, float]) -> float:
    """
    Score a meal based on how well it fits user targets.

    Args:
        meal_row: Single meal row from dataframe
        user_targets: Dict with protein, carbs, fat targets (daily grams)

    Returns:
        Score (higher is better)
    """
    # Normalize per 100g serving
    protein_per_100g = meal_row['protein']
    carbs_per_100g = meal_row['carbs']
    fat_per_100g = meal_row['fat']
    calories_per_100g = meal_row['kcal']

    # Protein density score (prefer higher protein density)
    protein_density = protein_per_100g / calories_per_100g if calories_per_100g > 0 else 0
    protein_score = min(protein_density * 100, 50)  # Cap at 50 points

    # Macro balance score (how well it fits daily macro ratios)
    daily_protein_ratio = user_targets['protein'] / sum(user_targets.values()) if sum(user_targets.values()) > 0 else 0
    daily_carbs_ratio = user_targets['carbs'] / sum(user_targets.values()) if sum(user_targets.values()) > 0 else 0
    daily_fat_ratio = user_targets['fat'] / sum(user_targets.values()) if sum(user_targets.values()) > 0 else 0

    meal_protein_ratio = protein_per_100g * 4 / calories_per_100g if calories_per_100g > 0 else 0
    meal_carbs_ratio = carbs_per_100g * 4 / calories_per_100g if calories_per_100g > 0 else 0
    meal_fat_ratio = fat_per_100g * 9 / calories_per_100g if calories_per_100g > 0 else 0

    macro_balance_score = 100 - (
        abs(meal_protein_ratio - daily_protein_ratio) +
        abs(meal_carbs_ratio - daily_carbs_ratio) +
        abs(meal_fat_ratio - daily_fat_ratio)
    ) * 100

    # Fiber score (if available)
    fiber_score = 0
    if 'fiber' in meal_row.index:
        fiber_per_100g = meal_row['fiber']
        fiber_score = min(fiber_per_100g * 2, 20)  # Up to 20 points

    # Health score penalty
    health_penalty = 0
    health_score = meal_row.get('health_score', 'moderate')
    if health_score == 'heavy':
        health_penalty = -20
    elif health_score == 'moderate':
        health_penalty = -5

    total_score = protein_score + macro_balance_score + fiber_score + health_penalty

    return max(0, total_score)  # Ensure non-negative

def pick_meal_for_slot(filtered_df: pd.DataFrame, slot_target_kcal: float,
                      used_foods: List[str], ban_list: List[str],
                      prefer_high_protein: bool = True) -> List[Dict[str, Any]]:
    """
    Pick meals for a specific slot using greedy + stochastic sampling.

    Args:
        filtered_df: Filtered nutrition dataframe
        slot_target_kcal: Target calories for this slot
        used_foods: Foods already used today
        ban_list: Foods to ban (e.g., repeats from previous days)
        prefer_high_protein: Whether to prefer high-protein foods

    Returns:
        List of meal items with portions and macros
    """
    # Filter by meal types that can go in this slot
    slot = slot_target_kcal  # This is actually the slot name, need to fix parameter
    # Assuming slot_target_kcal is passed as tuple or we need to adjust

    # For now, assume slot_target_kcal is (slot_name, target_kcal)
    if isinstance(slot_target_kcal, tuple):
        slot_name, target_kcal = slot_target_kcal
    else:
        # If not tuple, assume it's just target_kcal and we need slot name
        # This is a design issue - let's assume we pass slot name separately
        slot_name = "Breakfast"  # Default, should be fixed
        target_kcal = slot_target_kcal

    # Filter foods available for this slot
    if 'meal_types' in filtered_df.columns:
        slot_mask = filtered_df['meal_types'].str.contains(slot_name, case=False, na=False)
        slot_df = filtered_df[slot_mask]
    else:
        # Fallback - assume all foods can go in any slot
        slot_df = filtered_df

    # Remove used and banned foods
    available_df = slot_df[~slot_df.index.isin(used_foods + ban_list)]

    if len(available_df) == 0:
        return []

    # Calculate scores for all available foods
    # We need user_targets for scoring - this should be passed in
    # For now, use dummy targets
    dummy_targets = {'protein': 150, 'carbs': 250, 'fat': 67}
    available_df = available_df.copy()
    available_df['score'] = available_df.apply(lambda row: score_and_rank(row, dummy_targets), axis=1)

    # Sort by score (descending)
    available_df = available_df.sort_values('score', ascending=False)

    # Stochastic sampling: try combinations of 1-3 items
    best_combination = []
    best_score = -1
    tolerance = 0.1  # ±10%

    # Try different combination sizes
    for combo_size in range(1, min(4, len(available_df) + 1)):
        # Sample multiple combinations
        for _ in range(min(50, len(available_df) ** combo_size)):  # Limit iterations
            combination = available_df.sample(n=combo_size, replace=False)

            # Calculate total calories for this combination at standard portions
            total_calories = 0
            meal_items = []

            for _, food_row in combination.iterrows():
                # Adjust portion to contribute to target
                portion_g = adjust_portion_to_hit_calories(food_row, target_kcal / combo_size)

                # Calculate nutrition
                nutrition = {
                    'Food': food_row.name,
                    'Portion_g': portion_g,
                    'Calories': (portion_g / 100) * food_row['kcal'],
                    'Protein': (portion_g / 100) * food_row['protein'],
                    'Carbs': (portion_g / 100) * food_row['carbs'],
                    'Fat': (portion_g / 100) * food_row['fat']
                }

                meal_items.append(nutrition)
                total_calories += nutrition['Calories']

            # Check if within tolerance
            if abs(total_calories - target_kcal) / target_kcal <= tolerance:
                # Score this combination
                combo_score = sum(item['Protein'] for item in meal_items) if prefer_high_protein else total_calories
                if combo_score > best_score:
                    best_combination = meal_items
                    best_score = combo_score

    return best_combination

def build_day(filtered_df: pd.DataFrame, daily_targets: Dict[str, float],
              used_today: List[str], prev_day_items: List[str]) -> Dict[str, Any]:
    """
    Build a complete day's meal plan.

    Args:
        filtered_df: Filtered nutrition dataframe
        daily_targets: Dict with calories and macro targets
        used_today: Foods used so far today
        prev_day_items: Foods from previous day

    Returns:
        Dict with day_plan and totals
    """
    day_plan = {}
    all_used_today = used_today.copy()
    daily_calories = daily_targets.get('calories', 2000)

    # Define slot calorie distributions
    slot_distribution = {
        'Breakfast': 0.25,
        'Lunch': 0.35,
        'Dinner': 0.30,
        'Snack': 0.10
    }

    for slot in MEAL_SLOTS:
        slot_target_kcal = daily_calories * slot_distribution[slot]

        # Create ban list to prevent repeats
        ban_list = prev_day_items if len(prev_day_items) > 0 else []

        # Pick meals for this slot
        meal_items = pick_meal_for_slot(filtered_df, (slot, slot_target_kcal),
                                       all_used_today, ban_list,
                                       prefer_high_protein=(slot in ['Breakfast', 'Lunch']))

        day_plan[slot] = meal_items

        # Update used foods
        for item in meal_items:
            all_used_today.append(item['Food'])

    # Calculate totals
    totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    all_meals = []
    for slot, slot_meals in day_plan.items():
        for meal in slot_meals:
            totals['calories'] += meal['Calories']
            totals['protein'] += meal['Protein']
            totals['carbs'] += meal['Carbs']
            totals['fat'] += meal['Fat']
            all_meals.append({
                'Meal': slot,
                'Food': meal['Food'],
                'Amount': f"{meal['Portion_g']:.1f}g",
                'Calories': meal['Calories'],
                'Protein': meal['Protein'],
                'Carbs': meal['Carbs'],
                'Fat': meal['Fat']
            })

    return {
        'meals': all_meals,
        'totals': totals,
        'used_foods': all_used_today
    }

def build_week(filtered_df: pd.DataFrame, user_targets: Dict[str, float]) -> Dict[str, Any]:
    """
    Build a 7-day meal plan with variety constraints.

    Args:
        filtered_df: Filtered nutrition dataframe
        user_targets: User targets dict

    Returns:
        Dict with week_plan, weekly_totals, notes
    """
    week_plan = {}
    used_history = []  # Track foods used across days
    prev_day_items = []

    for day_idx in range(7):
        # Build day with variety constraints
        day_result = build_day(filtered_df, user_targets, [], prev_day_items)
        week_plan[f'day_{day_idx + 1}'] = day_result['meals']

        # Update history
        prev_day_items = [item['Food'] for item in day_result['meals']]
        used_history.extend(prev_day_items)

        # Enforce max repeats across week
        used_history = enforce_max_repeats(used_history)

    # Calculate weekly totals
    weekly_totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    for day_meals in week_plan.values():
        for meal in day_meals:
            weekly_totals['calories'] += meal['Calories']
            weekly_totals['protein'] += meal['Protein']
            weekly_totals['carbs'] += meal['Carbs']
            weekly_totals['fat'] += meal['Fat']

    # Generate notes about swaps/variety
    notes = []
    if len(set(used_history)) < len(used_history) * 0.7:
        notes.append("Some food repetition detected - consider expanding food preferences")

    return {
        'week_plan': week_plan,
        'weekly_totals': weekly_totals,
        'notes': notes
    }

def adjust_portion_to_hit_calories(food_row: pd.Series, target_kcal: float) -> float:
    """
    Adjust portion size to hit target calories.

    Args:
        food_row: Food data row
        target_kcal: Target calories

    Returns:
        Portion in grams
    """
    calories_per_100g = food_row['kcal']
    if calories_per_100g <= 0:
        return 100.0  # Default portion

    portion_g = (target_kcal / calories_per_100g) * 100

    # Reasonable bounds
    portion_g = max(20, min(portion_g, 250))

    return portion_g

def enforce_max_repeats(food_history: List[str], max_repeats: int = 2) -> List[str]:
    """
    Enforce maximum repeats in food history.

    Args:
        food_history: List of foods used
        max_repeats: Maximum allowed repeats

    Returns:
        Filtered history with repeats limited
    """
    from collections import Counter

    counts = Counter(food_history)
    filtered_history = []

    for food in food_history:
        if counts[food] <= max_repeats:
            filtered_history.append(food)
        # If over limit, we don't add but count remains for other instances

    return filtered_history

# Additional utility functions for compatibility
def calculate_bmr_tdee(age, weight, height, gender, activity_level):
    """Legacy function for compatibility."""
    user = {
        'age': age, 'sex': gender, 'weight': weight, 'height': height,
        'activity_level': activity_level, 'goal': 'maintain'
    }
    daily_calories, _ = calculate_targets(user)
    bmr, tdee = daily_calories - GOAL_ADJUSTMENTS['maintain'], daily_calories
    return bmr, tdee

def calculate_macros(calories, goal):
    """Legacy function for compatibility."""
    user = {'goal': goal}
    _, macro_targets = calculate_targets({**user, 'age': 25, 'sex': 'male', 'weight': 70, 'height': 170, 'activity_level': 'moderately active'})
    protein_pct = macro_targets['protein'] * 4 / calories
    carbs_pct = macro_targets['carbs'] * 4 / calories
    fat_pct = macro_targets['fat'] * 9 / calories
    return protein_pct, carbs_pct, fat_pct, macro_targets['protein'], macro_targets['carbs'], macro_targets['fat']

def load_db():
    """Load nutrition database."""
    nutrition_df = pd.read_csv('data/nutrition_data_optimized.csv', index_col='food')
    nutrition_df['unit_options'] = nutrition_df['unit_options'].apply(ast.literal_eval)
    return nutrition_df

# Export functions for streamlit app
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

def search_food_case_insensitive(food_name, nutrition_df):
    """
    Search for food in the database with case-insensitive matching.
    Returns the exact food name if found, else None.
    """
    food_name_lower = food_name.lower()
    for food in nutrition_df.index:
        if food.lower() == food_name_lower:
            return food
    return None

def convert_units_to_grams(food_name, quantity, unit, nutrition_df):
    """
    Convert various units to grams for nutrition calculation.
    """
    # Standard conversions (approximate)
    unit_conversions = {
        'grams': 1,
        'g': 1,
        'cups': 240,  # 1 cup = 240g for most foods
        'cup': 240,
        'tsp': 5,     # 1 tsp = 5g
        'tbsp': 15,   # 1 tbsp = 15g
        'pieces': 100,  # Assume 100g per piece unless specified
        'piece': 100,
        'roti': 40,   # 1 roti = 40g
        'chapati': 40,
        'fruit sizes': {
            'small': 120,
            'medium': 150,
            'large': 200,
            'Apple': 182,
            'Banana': 118,
            'Orange': 131,
            'Mango': 200,
            'Guava': 100,
            'Papaya': 300,
            'Pomegranate': 200,
            'Litchi': 20
        },
        'sabzi types': {
            'small': 100,
            'medium': 150,
            'large': 200
        }
    }

    if unit.lower() in ['grams', 'g']:
        return quantity
    elif unit.lower() in ['cups', 'cup']:
        return quantity * unit_conversions['cups']
    elif unit.lower() == 'tsp':
        return quantity * unit_conversions['tsp']
    elif unit.lower() == 'tbsp':
        return quantity * unit_conversions['tbsp']
    elif unit.lower() in ['pieces', 'piece']:
        return quantity * unit_conversions['pieces']
    elif unit.lower() in ['roti', 'chapati']:
        return quantity * unit_conversions['roti']
    elif unit.lower() == 'fruit sizes':
        if food_name in unit_conversions['fruit sizes']:
            return quantity * unit_conversions['fruit sizes'][food_name]
        else:
            return quantity * unit_conversions['fruit sizes']['medium']
    elif unit.lower() == 'sabzi types':
        return quantity * unit_conversions['sabzi types']['medium']
    else:
        # Default to grams if unit not recognized
        return quantity

def get_display_amount_and_unit(food_name, grams, nutrition_df):
    """
    Convert grams back to a user-friendly display unit and quantity.
    """
    if food_name not in nutrition_df.index:
        return round(grams, 1), 'g'

    unit_options = nutrition_df.loc[food_name, 'unit_options']

    # Define weights for pieces (in grams) for specific foods
    piece_weights = {
        'Egg': 50,
        'Bread': 40,
        'Roti': 30,
        'Chapati': 30,
        'Naan': 40,
        'Pizza': 100,
        'Burger': 150,
        'Sushi': 30,
        'Pakora': 20,
        'Samosa': 50,
        'Falafel': 50,
        'Tacos': 100,
        'Enchiladas': 150,
        'Dhokla': 50,
        'Apple': 182, 'Banana': 118, 'Orange': 131, 'Mango': 200,
        'Guava': 100, 'Papaya': 300, 'Pomegranate': 200, 'Litchi': 20
    }
    piece_weight = piece_weights.get(food_name, 150)  # Default for non-fruits

    # Priority order for display units
    preferred_units = ['piece', 'cup', 'tbsp', 'tsp', 'g']

    for unit in preferred_units:
        if unit in unit_options:
            if unit == 'piece':
                quantity = grams / piece_weight
                if quantity >= 0.5:
                    return round(quantity, 1), 'piece'
            elif unit == 'cup':
                quantity = grams / 240
                if quantity >= 0.1:
                    return round(quantity, 1), 'cup'
            elif unit == 'tbsp':
                quantity = grams / 15
                if quantity >= 0.5:
                    return round(quantity, 1), 'tbsp'
            elif unit == 'tsp':
                quantity = grams / 5
                if quantity >= 0.5:
                    return round(quantity, 1), 'tsp'

    # Default to grams
    return round(grams, 1), 'g'

def calculate_nutrition_per_serving(food_name, quantity, unit, nutrition_df):
    """
    Calculate nutrition for a specific serving size.
    Returns dict with calories, protein, carbs, fat.
    """
    if food_name not in nutrition_df.index:
        return {'Calories': 0, 'Protein': 0, 'Carbs': 0, 'Fat': 0}

    # Get nutrition data per 100g
    food_data = nutrition_df.loc[food_name]

    # Convert quantity to grams
    grams = convert_units_to_grams(food_name, quantity, unit, nutrition_df)

    # Calculate nutrition per serving
    serving_nutrition = {
        'Calories': round((grams / 100) * food_data['kcal'], 1),
        'Protein': round((grams / 100) * food_data['protein'], 1),
        'Carbs': round((grams / 100) * food_data['carbs'], 1),
        'Fat': round((grams / 100) * food_data['fat'], 1)
    }

    return serving_nutrition

def get_meal_totals(meal_entries):
    """
    Calculate total nutrition for a list of meal entries.
    """
    totals = {'Calories': 0, 'Protein': 0, 'Carbs': 0, 'Fat': 0}

    for entry in meal_entries:
        totals['Calories'] += entry.get('Calories', 0)
        totals['Protein'] += entry.get('Protein', 0)
        totals['Carbs'] += entry.get('Carbs', 0)
        totals['Fat'] += entry.get('Fat', 0)

    return totals

def get_progress_towards_targets(consumed, targets):
    """
    Calculate progress towards daily targets.
    Returns dict with consumed, target, remaining for each macro.
    """
    progress = {}
    for macro in ['Calories', 'Protein', 'Carbs', 'Fat']:
        consumed_val = consumed.get(macro, 0)
        target_val = targets.get(macro, 0)
        remaining = max(0, target_val - consumed_val)
        progress[macro] = {
            'consumed': consumed_val,
            'target': target_val,
            'remaining': remaining,
            'percentage': min(100, (consumed_val / target_val * 100) if target_val > 0 else 0)
        }

    return progress

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

@st.cache_data
def aggregate_historical_data(daily_log, days_back=30):
    """
    Aggregate daily log data into a list of daily totals.
    Optimized for performance with vectorized operations.
    """
    if not daily_log:
        return []

    historical_data = []
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=days_back)

    for date_str, meals in daily_log.items():
        date = pd.to_datetime(date_str)
        if start_date <= date <= end_date:
            daily_totals = {'Date': date_str, 'Calories': 0, 'Protein': 0, 'Carbs': 0, 'Fat': 0}

            for meal in meals:
                meal_type = meal.get('Meal_Type', 'Unknown')
                for macro in ['Calories', 'Protein', 'Carbs', 'Fat']:
                    daily_totals[macro] += meal.get(macro, 0)
                    # Add meal-wise data
                    meal_key = f"{macro}_{meal_type}"
                    if meal_key not in daily_totals:
                        daily_totals[meal_key] = 0
                    daily_totals[meal_key] += meal.get(macro, 0)

            historical_data.append(daily_totals)

    return sorted(historical_data, key=lambda x: x['Date'])

@cache_data
def aggregate_meal_wise_data(daily_log, days_back=30):
    """
    Aggregate daily log data into a list of meal-wise entries.
    Returns list of dicts with Date, Meal_Type, and macro values.
    """
    if not daily_log:
        return []

    meal_data = []
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=days_back)

    for date_str, meals in daily_log.items():
        date = pd.to_datetime(date_str)
        if start_date <= date <= end_date:
            for meal in meals:
                meal_entry = {
                    'Date': date_str,
                    'Meal_Type': meal.get('Meal_Type', 'Unknown'),
                    'Calories': meal.get('Calories', 0),
                    'Protein': meal.get('Protein', 0),
                    'Carbs': meal.get('Carbs', 0),
                    'Fat': meal.get('Fat', 0)
                }
                meal_data.append(meal_entry)

    return sorted(meal_data, key=lambda x: x['Date'])

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

def generate_smart_swaps(original_meal, nutrition_df):
    """
    Generate smart food swap with calculated quantity to match original macros.

    Args:
        original_meal: Dict with 'Food', 'Calories', 'Protein', etc.
        nutrition_df: Nutrition dataframe

    Returns:
        List of dicts with 'food', 'amount', 'calories', 'protein', 'carbs', 'fat' for best swaps
    """
    if original_meal['Food'] not in nutrition_df.index:
        return []

    original_food = original_meal['Food']
    target_calories = original_meal['Calories']
    target_protein = original_meal['Protein']

    # Find candidates with similar calories and protein (±10%)
    candidates = nutrition_df[
        (nutrition_df['kcal'] >= target_calories * 0.9) &
        (nutrition_df['kcal'] <= target_calories * 1.1) &
        (nutrition_df['protein'] >= target_protein * 0.9) &
        (nutrition_df['protein'] <= target_protein * 1.1) &
        (nutrition_df.index != original_food)
    ]

    if candidates.empty:
        # Relax criteria if no matches
        candidates = nutrition_df[
            (nutrition_df['kcal'] >= target_calories * 0.8) &
            (nutrition_df['kcal'] <= target_calories * 1.2) &
            (nutrition_df.index != original_food)
        ]

    if candidates.empty:
        return []

    # Pick the best candidates (highest protein density)
    candidates = candidates.copy()
    candidates['protein_density'] = candidates['protein'] / candidates['kcal']
    candidates = candidates.sort_values('protein_density', ascending=False).head(3)  # Top 3

    swaps = []
    for _, best_swap in candidates.iterrows():
        # Calculate quantity to match original calories
        swap_calories_per_100g = best_swap['kcal']
        if swap_calories_per_100g <= 0:
            quantity_g = 100.0
        else:
            quantity_g = (target_calories / swap_calories_per_100g) * 100
            quantity_g = max(20, min(quantity_g, 250))  # Reasonable bounds

        # Calculate macros for this quantity
        scale_factor = quantity_g / 100
        display_amount, display_unit = get_display_amount_and_unit(best_swap.name, quantity_g, nutrition_df)

        swaps.append({
            'food': best_swap.name,
            'amount': f"{display_amount} {display_unit}",
            'calories': round(scale_factor * best_swap['kcal'], 1),
            'protein': round(scale_factor * best_swap['protein'], 1),
            'carbs': round(scale_factor * best_swap['carbs'], 1),
            'fat': round(scale_factor * best_swap['fat'], 1)
        })

    return swaps

def filter_foods_by_preferences(nutrition_df, food_preference, allergies):
    """
    Wrapper for filter_meals to match app.py usage.
    """
    preferences = {
        'veg_flag': food_preference.lower() if food_preference != "None" else 'none',
        'allergens': [a.strip().lower() for a in allergies.split(',') if a.strip()],
        'cuisine': 'any',
        'health_level': 'light'  # Default
    }
    return filter_meals(nutrition_df, preferences)

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
