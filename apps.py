from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import h5py
import os
import logging
from sklearn.preprocessing import OneHotEncoder
from google.cloud import secretmanager, storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Set the environment variable for Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./serviceKey.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "capstone-playground-423804"

# Function to retrieve a secret from Secret Manager
def get_secret(secret_id):
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            raise ValueError("Environment variable GOOGLE_CLOUD_PROJECT is not set.")
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(name=name)
        secret_data = response.payload.data  # Do not decode as 'UTF-8'
        logging.info(f"Retrieved secret {secret_id}")
        return secret_data
    except Exception as e:
        logging.error(f"Failed to access secret {secret_id}: {e}")
        raise

# Function to download a file from Google Cloud Storage
def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")
    except Exception as e:
        logging.error(f"Failed to download file from GCS: {e}")
        raise

# Load secrets from Secret Manager
try:
    model_path_secret = get_secret('MODEL_H5_FILE_PATH')
    csv_path_secret = get_secret('MODEL_CSV_FILE_PATH')
except Exception as e:
    logging.error(f"Failed to load secret: {e}")
    raise

# Assume the secret is binary data representing the path to the model
model_path_binary = model_path_secret
csv_path_binary = csv_path_secret

# Local paths for downloaded files
local_model_path = './model/Recommended_Sugar_Intake_Model.h5'
local_csv_path = './model/Additional_Model.csv'
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)

# Save the binary secret data to a local file
with open(local_model_path, 'wb') as f:
    f.write(model_path_binary)

with open(local_csv_path, 'wb') as f:
    f.write(csv_path_binary)

# Parse the GCS paths from the secret binary data (assuming the secret data is paths)
model_bucket_name, model_blob_name = model_path_secret.replace("gs://", "").split("/", 1)
csv_bucket_name, csv_blob_name = csv_path_secret.replace("gs://", "").split("/", 1)

# Download the model and CSV files from GCS
try:
    download_file_from_gcs(model_bucket_name, model_blob_name, local_model_path)
    download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)
except Exception as e:
    logging.error(f"Failed to download files from GCS: {e}")
    raise

# Load the model from .h5 file
def load_model_from_h5(model_path):
    try:
        with h5py.File(model_path, 'r') as h5file:
            model = {
                'coefficients': np.array(h5file['linear_regression_coefficients']),
                'intercept': np.array(h5file['linear_regression_intercept']),
                'numeric_features': list(h5file['numeric_features']),
                'categorical_features': list(h5file['categorical_features']),
                'categories': [np.array(h5file[f'category_{i}']) for i in range(len(h5file.keys()) - 4)]
            }
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from h5 file: {e}")
        raise

# Load the CSV model
def load_model_from_csv(csv_path):
    try:
        model = pd.read_csv(csv_path)
        logging.info(f"Loaded CSV model from {csv_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load CSV model: {e}")
        raise

# Function to perform prediction using loaded models
def predict_sugar_intake(model_h5, model_csv, age, height, weight, diabetes_history, diabetes_heritage):
    try:
        numeric_features = model_h5['numeric_features']
        categorical_features = model_h5['categorical_features']
        categories = model_h5['categories']
        
        # Preprocess input data
        input_data = {
            'age': age,
            'height': height,
            'weight': weight,
            'diabetes_history': diabetes_history,
            'diabetes_heritage': diabetes_heritage
        }
        
        # Convert categorical variables to one-hot encoding
        encoded_features = []
        for cat_feature, category in zip(categorical_features, categories):
            encoder = OneHotEncoder(categories=[category], handle_unknown='ignore')
            encoded_value = encoder.fit_transform([[input_data[cat_feature]]]).toarray()[0]
            encoded_features.extend(encoded_value)
        
        # Combine numeric and encoded categorical features
        processed_features = [input_data[feat] for feat in numeric_features]
        processed_features.extend(encoded_features)
        
        # Perform prediction using the .h5 model
        predicted_sugar_intake_h5 = sum([coef * feat for coef, feat in zip(model_h5['coefficients'], processed_features)]) + model_h5['intercept']
        
        # Perform prediction using the CSV model
        # Assuming the CSV model contains additional coefficients
        additional_coefficients = model_csv['coefficients'].values
        additional_intercept = model_csv['intercept'].values[0]
        predicted_sugar_intake_csv = sum([coef * feat for coef, feat in zip(additional_coefficients, processed_features)]) + additional_intercept
        
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
    except Exception as e:
        logging.error(f"Error in /predictRSI: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
