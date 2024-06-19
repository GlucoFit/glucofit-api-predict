import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import os
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Path to your .h5 file
local_model_path = './model/Recommended_Sugar_Intake_Model_TF.h5'

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

# Function to create and fit label encoders from a DataFrame
def create_label_encoders(data, categorical_features):
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data[feature])
        label_encoders[feature] = le
    return label_encoders

# Load mock data from CSV to fit label encoders
mock_data_path = './csv/Dummy_data_no_user.csv'
mock_data = pd.read_csv(mock_data_path)

categorical_features = ['diabetes_history', 'diabetes_heritage']
numeric_features = ['age', 'height', 'weight']
label_encoders = create_label_encoders(mock_data, categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Fit the preprocessor with mock data
preprocessor.fit(mock_data)

# Function to perform prediction using loaded model
def predict_sugar_intake(model, age, height, weight, diabetes_history, diabetes_heritage):
    try:
        # Convert input data to correct types
        age = float(age)
        height = float(height)
        weight = float(weight)
        diabetes_history = str(diabetes_history)
        diabetes_heritage = str(diabetes_heritage)

        # Preprocess input data
        input_data = pd.DataFrame({
            'age': [age],
            'height': [height],
            'weight': [weight],
            'diabetes_history': [diabetes_history],
            'diabetes_heritage': [diabetes_heritage]
        })

        logging.info(f"Input data before label encoding: {input_data}")

        # Apply label encoding to categorical features
        # for feature in categorical_features:
        #     input_data[feature] = label_encoders[feature].transform(input_data[feature])

        logging.info(f"Input data after label encoding: {input_data}")

        # Apply the preprocessor to the input data
        input_data_processed = preprocessor.transform(input_data)

        # Check for NaN values
        if np.isnan(input_data_processed).any():
            logging.error(f"NaN values found in processed input data: {input_data_processed}")
            raise ValueError("NaN values found in processed input data")

        logging.info(f"Processed input data (type): {type(input_data_processed)}")
        logging.info(f"Processed input data (dtype): {input_data_processed.dtype}")
        logging.info(f"Processed input data: {input_data_processed}")

        # Ensure input_data_processed is a float32 numpy array
        input_data_processed = input_data_processed.astype(np.float32)

        # Predict the recommended sugar intake
        predicted_sugar_intake = model.predict(input_data_processed)
        logging.info(f"Predicted sugar intake: {predicted_sugar_intake}")

        return predicted_sugar_intake[0][0].item()

    except Exception as e:
        logging.error(f"Failed to predict sugar intake: {e}")
        raise

# Route for predicting recommended sugar intake
@app.route('/predictRSI', methods=['POST'])
def predict_recommended_sugar_intake():
    try:
        data = request.json
        required_fields = ['age', 'height', 'weight', 'diabetes_history', 'diabetes_heritage']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field: {field}"}), 400

        age = data['age']
        height = data['height']
        weight = data['weight']
        diabetes_history = data['diabetes_history']
        diabetes_heritage = data['diabetes_heritage']

        # Load model from .h5 file
        model_h5 = load_model_from_h5(local_model_path)

        # Perform prediction
        predicted_sugar_intake = predict_sugar_intake(model_h5, age, height, weight, diabetes_history, diabetes_heritage)

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
