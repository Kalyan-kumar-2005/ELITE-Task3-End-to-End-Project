# app.py

from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
import pandas as pd

# Load the model and feature names
try:
    model = load('model.joblib')
    feature_names = load('model_features.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: model.joblib or model_features.joblib not found. Run model_builder.py first.")
    # Exit if the model isn't available
    exit()

# Initialize the Flask application
app = Flask(__name__)

# 1. Route for the Home Page (The Web Interface)
@app.route('/')
def home():
    # This will load the index.html file (Step 3)
    return render_template('index.html')

# 2. Route for Prediction (Handles the data submission)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the web form submission
        data = request.form.to_dict()
        
        # Prepare the data for the model
        # We assume the form fields match our feature_names: 'SqFt', 'Bedrooms', 'Age'
        input_data = [float(data[feature]) for feature in feature_names]
        
        # Convert to a format the model expects (DataFrame or numpy array)
        final_features = pd.DataFrame([input_data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # The model predicts the price (in thousands)
        output = round(prediction[0], 2)
        
        # Return the result back to the HTML page
        return render_template('index.html', 
                               prediction_text=f'Estimated House Price: ${output:,.2f} Thousand',
                               features=data)
                               
    except Exception as e:
        # Handle errors gracefully (e.g., if a user enters text instead of a number)
        return render_template('index.html', 
                               prediction_text=f'Error: Invalid input data. Please enter numbers.',
                               error_detail=str(e))

if __name__ == "__main__":
    # Flask will start on your local machine, usually at http://127.0.0.1:5000
    print("Starting Flask web server...")
    app.run(debug=True) # debug=True allows for easy testing