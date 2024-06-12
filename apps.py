from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import requests
import os

app = Flask(__name__)

# Google Cloud Storage credentials
model_url = "https://storage.googleapis.com/sugar_intake_model-01/Recommended_Sugar_Intake_Model.h5"

# Load the model
def load_model_from_url(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        model_file_path = "./Recommended_Sugar_Intake_Model.h5"
        with open(model_file_path, 'wb') as f:
            f.write(response.content)
        model = tf.keras.models.load_model(model_file_path)
        os.remove(model_file_path)  # Remove the temporary model file
        return model
    else:
        raise Exception("Failed to download model")

model = load_model_from_url(model_url)

def preprocess_input(file):
    # Assuming the CSV file contains columns: 'age', 'height', 'weight', 'diabetes_history', 'diabetes_heritage'
    df = pd.read_csv(file)
    # Implement any additional preprocessing if needed
    return df

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    input_data = preprocess_input(file)

    # Assuming your model expects a specific input format
    predictions = model.predict(input_data)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
