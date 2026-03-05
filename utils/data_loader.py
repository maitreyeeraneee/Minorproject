import pandas as pd
import ast
from typing import Dict, List, Tuple, Optional, Any

def load_db():
    """Load nutrition database."""
    nutrition_df = pd.read_csv('data/nutrition_data_optimized.csv', index_col='food')
    nutrition_df['unit_options'] = nutrition_df['unit_options'].apply(ast.literal_eval)
    return nutrition_df

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
