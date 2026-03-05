# Activity level multipliers for TDEE calculation
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly active': 1.375,
    'moderately active': 1.55,
    'very active': 1.725,
    'extra active': 1.9
}

# Macronutrient ratios by goal
MACRO_RATIOS = {
    'lose weight': {'protein': 0.35, 'carbs': 0.40, 'fat': 0.25},
    'gain muscle': {'protein': 0.30, 'carbs': 0.50, 'fat': 0.20},
    'maintain': {'protein': 0.25, 'carbs': 0.50, 'fat': 0.25}
}

# BMI categories
BMI_CATEGORIES = {
    'underweight': (0, 18.5),
    'normal': (18.5, 25),
    'overweight': (25, 30),
    'obese': (30, float('inf'))
}

# Meal types
MEAL_TYPES = ['Breakfast', 'Lunch', 'Dinner', 'Snack']

# Default targets
DEFAULT_TARGETS = {
    'calories': 2000,
    'protein': 150,
    'carbs': 250,
    'fat': 67
}
