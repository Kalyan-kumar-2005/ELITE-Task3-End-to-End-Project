# model_builder.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump # Tool to save the model (like pickle)
import numpy as np

def build_and_save_model():
    print("--- 1. Building and Saving the Model ---")
    
    # 1. DATA COLLECTION & PREPROCESSING
    # Note: We'll use a sample dataset here. In a real project, you'd load a CSV.
    # We will simulate a small house price dataset for simplicity.
    data = {
        'SqFt': [1000, 1500, 2000, 800, 1200, 1800, 2500, 900],
        'Bedrooms': [2, 3, 4, 1, 2, 3, 5, 2],
        'Age': [20, 10, 5, 30, 15, 8, 2, 25],
        'Price': [200, 300, 450, 150, 250, 400, 550, 180] # Price in thousands
    }
    df = pd.DataFrame(data)
    
    # Simple Preprocessing (No heavy work needed for this small example)
    X = df[['SqFt', 'Bedrooms', 'Age']]
    y = df['Price']
    
    # 2. MODEL DEVELOPMENT (Training)
    
    # Split the data (optional for simple examples, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Check the model score
    score = model.score(X_test, y_test)
    print(f"Model trained successfully. Test R-squared score: {score:.2f}")

    # 3. SAVE THE MODEL (Crucial for Deployment)
    # The 'joblib' library saves the trained model so Flask can load it later.
    dump(model, 'model.joblib') 
    
    # Also save the feature names (columns) for Flask
    dump(list(X.columns), 'model_features.joblib') 
    
    print("Model and features saved as 'model.joblib' and 'model_features.joblib'.")

if __name__ == '__main__':
    build_and_save_model()