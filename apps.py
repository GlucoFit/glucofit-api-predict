from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import os

app = Flask(__name__)

# Google Cloud Storage credentials
model_url = "https://storage.googleapis.com/sugar_intake_model-01/Recommended_Sugar_Intake_Model.h5"

# Load the model
def load_model_from_url(model_url):
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad status codes
        model_file_path = "Recommended_Sugar_Intake_Model.h5"
        with open(model_file_path, 'wb') as f:
            f.write(response.content)
        model = joblib.load(model_file_path)
        os.remove(model_file_path)  # Remove the temporary model file
        return model
    except requests.exceptions.RequestException as e:
        print(f"Failed to download model. HTTP Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")
        print(f"Error: {e}")
        raise Exception("Failed to download model")
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        raise Exception("Failed to load model")

model = load_model_from_url(model_url)

def preprocess_input(file):
    try:
        df = pd.read_csv(file)
        # Implement any additional preprocessing if needed
        expected_columns = ['age', 'height', 'weight', 'diabetes_history', 'diabetes_heritage']
        if not all(column in df.columns for column in expected_columns):
            raise ValueError(f"CSV file must contain columns: {expected_columns}")
        return df
    except Exception as e:
        print(f"Error processing input file. Error: {e}")
        raise ValueError("Error processing input file")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        input_data = preprocess_input(file)

        # Assuming your model expects a specific input format
        predictions = model.predict(input_data)

        return jsonify({'predictions': predictions.tolist()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error during prediction. Error: {e}")
        return jsonify({'error': 'Unexpected error during prediction'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
