from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
import logging
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Local paths for model files
local_model_path = './model/Recommended_Sugar_Intake_Model.h5'
local_csv_path = './model/Additional_Model.csv'
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)

# Load the model from .h5 file
def load_model_from_h5(model_path):
    try:
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist.")
        model = load_model(model_path)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from h5 file: {e}")
        raise

# Load the CSV model
def load_model_from_csv(csv_path):
    try:
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file {csv_path} does not exist.")
        model = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV model from {csv_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load CSV model: {e}")
        raise

# Function to perform prediction using loaded models
def predict_sugar_intake(model_h5, model_csv, age, height, weight, diabetes_history, diabetes_heritage):
    try:
        # Validate data types
        if not isinstance(age, (int, float)):
            raise ValueError("Invalid data type for 'age'. Expected int or float.")
        if not isinstance(height, (int, float)):
            raise ValueError("Invalid data type for 'height'. Expected int or float.")
        if not isinstance(weight, (int, float)):
            raise ValueError("Invalid data type for 'weight'. Expected int or float.")
        if not isinstance(diabetes_history, (int, float)):
            raise ValueError("Invalid data type for 'diabetes_history'. Expected int or float.")
        if not isinstance(diabetes_heritage, (int, float)):
            raise ValueError("Invalid data type for 'diabetes_heritage'. Expected int or float.")
        
        # Preprocess input data
        # Update the input_data array to include the correct number of features (7 in this case)
        # Here, we add placeholders for the missing features (e.g., feature1 and feature2)
        feature1 = 0.0  # Placeholder value for additional feature 1
        feature2 = 0.0  # Placeholder value for additional feature 2
        input_data = np.array([[age, height, weight, diabetes_history, diabetes_heritage, feature1, feature2]])
        
        # Perform prediction using the .h5 model
        predicted_sugar_intake_h5 = model_h5.predict(input_data)[0]
        
        # Perform prediction using the CSV model
        # Assuming the CSV model contains additional coefficients
        additional_coefficients = model_csv['coefficients'].values
        additional_intercept = model_csv['intercept'].values[0]
        predicted_sugar_intake_csv = np.dot(input_data[0], additional_coefficients) + additional_intercept
        
        # Combine predictions (e.g., averaging the results)
        combined_prediction = (predicted_sugar_intake_h5 + predicted_sugar_intake_csv) / 2
        
        logging.info(f"Combined predicted sugar intake: {combined_prediction}")
        return combined_prediction
    except Exception as e:
        logging.error(f"Failed to predict sugar intake: {e}")
        raise

# Route for predicting recommended sugar intake
@app.route('/predictRSI', methods=['POST'])
def predict_recommended_sugar_intake():
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['age', 'height', 'weight', 'diabetes_history', 'diabetes_heritage']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field: {field}"}), 400
            try:
                # Try to convert the field to float
                data[field] = float(data[field])
            except ValueError:
                return jsonify({'error': f"Invalid data type for '{field}'. Expected int or float."}), 400
        
        age = data['age']
        height = data['height']
        weight = data['weight']
        diabetes_history = data['diabetes_history']
        diabetes_heritage = data['diabetes_heritage']
        
        # Load models from .h5 and .csv files
        model_h5 = load_model_from_h5(local_model_path)
        model_csv = load_model_from_csv(local_csv_path)
        
        # Perform prediction
        predicted_sugar_intake = predict_sugar_intake(model_h5, model_csv, age, height, weight, diabetes_history, diabetes_heritage)
        
        # Return prediction as JSON response
        return jsonify({'predicted_sugar_intake': predicted_sugar_intake}), 200
    
    except KeyError as e:
        logging.error(f"Missing key in input data: {e}")
        return jsonify({'error': f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        logging.error(f"Invalid data type in input data: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error in /predictRSI: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
