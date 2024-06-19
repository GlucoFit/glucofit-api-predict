import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import os
import logging
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Local paths for model files
local_model_path = './model/recomendation_sugar_intake-2.h5'
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

def create_label_encoders(data, categorical_features):
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data[feature])
        label_encoders[feature] = le
    return label_encoders

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

# Function to perform prediction using loaded model
def predict_sugar_intake(model, age, height, weight, diabetes_history, diabetes_heritage):
    try:
        # Validate data types
        # if not isinstance(age, (int, float)):
        #     raise ValueError("Invalid data type for 'age'. Expected int or float.")
        # if not isinstance(height, (int, float)):
        #     raise ValueError("Invalid data type for 'height'. Expected int or float.")
        # if not isinstance(weight, (int, float)):
        #     raise ValueError("Invalid data type for 'weight'. Expected int or float.")
        # if not isinstance(diabetes_history, (int, float)):
        #     raise ValueError("Invalid data type for 'diabetes_history'. Expected int or float.")
        # if not isinstance(diabetes_heritage, (int, float)):
        #     raise ValueError("Invalid data type for 'diabetes_heritage'. Expected int or float.")
        
        categorical_features = ['diabetes_history', 'diabetes_heritage']
        numeric_features = ['age', 'height', 'weight']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        print(diabetes_history)
        print("str")

        # Preprocess input data
        input_data = pd.DataFrame({
            'age': [age],
            'height': [height],
            'weight': [weight],
            'diabetes_history': [diabetes_history],
            'diabetes_heritage': [diabetes_heritage]
        })

        local_csv_path = './csv/Dummy_data_no_user.csv'
        mock_data = pd.read_csv(local_csv_path)

        label_encoders = create_label_encoders(mock_data, categorical_features)

        for feature in categorical_features:
            input_data[feature] = label_encoders[feature].transform(input_data[feature])
        # Apply the preprocessor to the input data
        input_data_processed = preprocessor.transform(input_data)

        # Predict the recommended sugar intake
        predicted_sugar_intake = model.predict(input_data_processed)

        return predicted_sugar_intake[0][0]
        
    #     logging.info(f"Predicted sugar intake: {predicted_sugar_intake}")
    #     return predicted_sugar_intake
    except Exception as e:
        logging.error(f"Failed to predict sugar intake: {e}")
        raise

# Route for predicting recommended sugar intake
@app.route('/predictRSI', methods=['POST'])
def predict_recommended_sugar_intake():
    # try:
        print("halo")
        data = request.json
        print(data)
        # Validate input data
        required_fields = ['age', 'height', 'weight', 'diabetes_history', 'diabetes_heritage']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field: {field}"}), 400
            # try:
            #     # Try to convert the field to float
            #     data[field] = float(data[field])
            # except ValueError:
            #     return jsonify({'error': f"Invalid data type for '{field}'. Expected int or float."}), 400
        
        age = data['age']
        height = data['height']
        weight = data['weight']
        diabetes_history = data['diabetes_history']
        diabetes_heritage = data['diabetes_heritage']
        
        # Load model from .h5 file
        print("string string before loading model")
        model_h5 = tf.keras.models.load_model('model/Recommended_Sugar_Intake_Model_TF.h5')
        print("string string after loading model")

        # Perform prediction
        predicted_sugar_intake = predict_sugar_intake(model_h5, age, height, weight, diabetes_history, diabetes_heritage)
        
        # Return prediction as JSON response
        return jsonify({'predicted_sugar_intake': predicted_sugar_intake.tolist()}), 200
    
    # except KeyError as e:
    #     logging.error(f"Missing key in input data: {e}")
    #     return jsonify({'error': f"Missing key: {str(e)}"}), 400
    # except ValueError as e:
    #     logging.error(f"Invalid data type in input data: {e}")
    #     return jsonify({'error': str(e)}), 400
    # except Exception as e:
    #     logging.error(f"Error in /predictRSI: {e}")
    #     return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
