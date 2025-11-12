import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_calorie_model():
    """
    Train a simple linear regression model to predict calories based on macros.
    This is optional and for demonstration purposes.
    """
    # Load nutrition data
    nutrition_df = pd.read_csv('data/nutrition_data.csv')
    
    # Prepare features (protein, carbs, fat) and target (calories)
    X = nutrition_df[['Protein', 'Carbs', 'Fat']]
    y = nutrition_df['Calories']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse:.2f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/calorie_predictor.pkl')
    
    return model

def predict_calories(protein, carbs, fat):
    """
    Predict calories using the trained model.
    """
    try:
        model = joblib.load('models/calorie_predictor.pkl')
        prediction = model.predict([[protein, carbs, fat]])[0]
        return max(0, prediction)  # Ensure non-negative
    except FileNotFoundError:
        # Fallback to simple calculation if model not available
        return (protein * 4) + (carbs * 4) + (fat * 9)

if __name__ == "__main__":
    train_calorie_model()
