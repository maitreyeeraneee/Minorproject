import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from .data_loader import get_display_amount_and_unit

# Constants
MEAL_SLOTS = ['Breakfast', 'Lunch', 'Dinner']  # Removed 'Snack' as default
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
        slot_target_kcal: Target calories for this slot (passed as tuple: (slot_name, target_kcal))
        used_foods: Foods already used today
        ban_list: Foods to ban (e.g., repeats from previous days)
        prefer_high_protein: Whether to prefer high-protein foods

    Returns:
        List of meal items with portions and macros
    """
    # Unpack slot name and target calories
    if isinstance(slot_target_kcal, tuple):
        slot_name, target_kcal = slot_target_kcal
    else:
        slot_name = "Breakfast"  # Default fallback
        target_kcal = slot_target_kcal

    # Filter foods available for this slot based on meal_types column
    if 'meal_types' in filtered_df.columns:
        # meal_types contains Python list strings like "['Breakfast', 'Snack']"
        # Parse the string and check if slot_name is in the list
        def meal_type_matches(meal_types_str):
            try:
                # Parse the string representation of a list
                import ast
                meal_list = ast.literal_eval(meal_types_str)
                return slot_name in meal_list
            except (ValueError, SyntaxError):
                # Fallback to string contains if parsing fails
                return slot_name in meal_types_str

        slot_df = filtered_df[filtered_df['meal_types'].apply(meal_type_matches)]
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

    # Stochastic sampling: prioritize combinations of 2-3 items
    best_combination = []
    best_score = -1
    tolerance = 0.15  # ±15% tolerance for better success rate

    # Prioritize 2-3 item combinations
    preferred_sizes = [2, 3, 1]  # Try 2 items first, then 3, then 1

    for combo_size in preferred_sizes:
        if combo_size >= len(available_df):
            continue

        # Sample multiple combinations for this size
        num_samples = min(100, max(10, len(available_df) * 2))  # More samples for better results
        for _ in range(num_samples):
            combination = available_df.sample(n=combo_size, replace=False)

            # Calculate total calories for this combination
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
                # Score this combination (prefer protein for main meals, calories for snacks)
                combo_score = sum(item['Protein'] for item in meal_items) if prefer_high_protein else total_calories
                if combo_score > best_score:
                    best_combination = meal_items
                    best_score = combo_score

        # If we found a good combination for this size, use it
        if best_combination:
            break

    # If no combination found, try with relaxed constraints
    if not best_combination and len(available_df) > 0:
        # Fallback: just pick top 2-3 items by score
        fallback_size = min(3, len(available_df))
        top_items = available_df.head(fallback_size)

        for _, food_row in top_items.iterrows():
            portion_g = adjust_portion_to_hit_calories(food_row, target_kcal / fallback_size)
            nutrition = {
                'Food': food_row.name,
                'Portion_g': portion_g,
                'Calories': (portion_g / 100) * food_row['kcal'],
                'Protein': (portion_g / 100) * food_row['protein'],
                'Carbs': (portion_g / 100) * food_row['carbs'],
                'Fat': (portion_g / 100) * food_row['fat']
            }
            best_combination.append(nutrition)

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

    # Define slot calorie distributions (no Snack by default)
    slot_distribution = {
        'Breakfast': 0.25,
        'Lunch': 0.40,
        'Dinner': 0.35
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

    # Reasonable bounds - cap at 250g for realism
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

def filter_foods_by_preferences(nutrition_df, food_preference, allergies, cuisine_preference=None):
    """
    Wrapper for filter_meals to match app.py usage.
    """
    preferences = {
        'veg_flag': food_preference.lower() if food_preference != "None" else 'none',
        'allergens': [a.strip().lower() for a in allergies.split(',') if a.strip()],
        'cuisine': cuisine_preference.lower() if cuisine_preference and cuisine_preference != "Any" else 'any',
        'health_level': 'light'  # Default
    }
    return filter_meals(nutrition_df, preferences)

def generate_smart_swaps(original_meal, nutrition_df):
    """
    Generate smart food swap with calculated quantity to match original macros.

    Args:
        original_meal: Dict with 'Food', 'Calories', 'Protein', etc.
        nutrition_df: Nutrition dataframe

    Returns:
        List of dicts with 'food', 'amount', 'calories', 'protein', 'carbs', 'fat' for best swaps
    """
    if isinstance(original_meal, str):
        # Handle string input for backward compatibility
        food = original_meal
        if food not in nutrition_df.index:
            return []
        target_calories = nutrition_df.loc[food, 'kcal']
        target_protein = nutrition_df.loc[food, 'protein']
    else:
        # Handle dict input
        food = original_meal.get('Food', '')
        target_calories = original_meal.get('Calories', 0)
        target_protein = original_meal.get('Protein', 0)

    if food not in nutrition_df.index:
        return []

    # Find candidates with similar calories and protein (±10%)
    candidates = nutrition_df[
        (nutrition_df['kcal'] >= target_calories * 0.9) &
        (nutrition_df['kcal'] <= target_calories * 1.1) &
        (nutrition_df['protein'] >= target_protein * 0.9) &
        (nutrition_df['protein'] <= target_protein * 1.1) &
        (nutrition_df.index != food)
    ]

    if candidates.empty:
        # Relax criteria if no matches
        candidates = nutrition_df[
            (nutrition_df['kcal'] >= target_calories * 0.8) &
            (nutrition_df['kcal'] <= target_calories * 1.2) &
            (nutrition_df.index != food)
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
