# Import functions from various utils modules to make them available at package level

# From data_loader.py
from .data_loader import (
    load_db, search_food_case_insensitive, convert_units_to_grams,
    get_display_amount_and_unit, calculate_nutrition_per_serving,
    get_meal_totals, get_progress_towards_targets,
    aggregate_historical_data, aggregate_meal_wise_data
)

# From meal_generator.py
from .meal_generator import (
    calculate_bmr_tdee, calculate_macros, filter_foods_by_preferences,
    generate_smart_swaps, filter_meals, build_week, build_day
)

# From ui_components.py
from .ui_components import (
    create_macro_chart, create_micronutrient_chart, get_random_tips,
    calculate_adaptive_recommendations, calculate_bmi, create_daily_chart,
    create_weekly_chart, create_monthly_chart, create_meal_wise_chart,
    create_daily_calorie_bar_chart, create_meal_wise_macro_bar_chart
)

# From utils.py (analytics functions)
from .analytics import (
    calculate_ideal_body_weight, calculate_body_fat_percentage,
    calculate_lean_body_mass, calculate_protein_target,
    calculate_daily_water_intake, calculate_calorie_balance,
    calculate_macro_ratios_string
)

# Make all functions available at package level
__all__ = [
    # data_loader functions
    'load_db', 'search_food_case_insensitive', 'convert_units_to_grams',
    'get_display_amount_and_unit', 'calculate_nutrition_per_serving',
    'get_meal_totals', 'get_progress_towards_targets',
    'aggregate_historical_data', 'aggregate_meal_wise_data',

    # meal_generator functions
    'calculate_bmr_tdee', 'calculate_macros', 'filter_foods_by_preferences',
    'generate_smart_swaps', 'filter_meals', 'build_week', 'build_day',

    # ui_components functions
    'create_macro_chart', 'create_micronutrient_chart', 'get_random_tips',
    'calculate_adaptive_recommendations', 'calculate_bmi', 'create_daily_chart',
    'create_weekly_chart', 'create_monthly_chart', 'create_meal_wise_chart',
    'create_daily_calorie_bar_chart', 'create_meal_wise_macro_bar_chart',

    # analytics functions
    'calculate_ideal_body_weight', 'calculate_body_fat_percentage',
    'calculate_lean_body_mass', 'calculate_protein_target',
    'calculate_daily_water_intake', 'calculate_calorie_balance',
    'calculate_macro_ratios_string', 'calculate_weekly_summary_stats',
    'create_weekly_nutrient_bar_chart', 'create_daily_calorie_pie_chart',
    'create_protein_line_plot'
]
