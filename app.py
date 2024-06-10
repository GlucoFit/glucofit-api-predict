from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from google.cloud import secretmanager, storage
import os

# Todo remove ini bang, bad practice
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './serviceKey.json'
os.makedirs('./model', exist_ok=True)
os.makedirs('./csv', exist_ok=True)

app = Flask(__name__)

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    # project_id = 'capstone-playground-423804'
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode('UTF-8')

def download_file_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

# Load secrets
model_path_secret = get_secret('DEEP_LEARNING_CBF_MODEL_PATH')
menu_csv_path_secret = get_secret('MENU_CSV_PATH')
user_csv_path_secret = get_secret('USER_CSV_PATH')

# Parse the GCS paths
model_bucket_name, model_blob_name = model_path_secret.replace("gs://", "").split("/", 1)
menu_bucket_name, menu_blob_name = menu_csv_path_secret.replace("gs://", "").split("/", 1)
user_bucket_name, user_blob_name = user_csv_path_secret.replace("gs://", "").split("/", 1)

# Local paths for downloaded files
local_model_path = './model/Deep_content_based_model.h5'
local_menu_csv_path = './csv/menu.csv'
local_user_csv_path = './csv/user.csv'

# Download the model and CSV files from GCS
download_file_from_gcs(model_bucket_name, model_blob_name, local_model_path)
download_file_from_gcs(menu_bucket_name, menu_blob_name, local_menu_csv_path)
download_file_from_gcs(user_bucket_name, user_blob_name, local_user_csv_path)

# Load the model
model = tf.keras.models.load_model(local_model_path)

# Load the encoders and preprocessors
ohe_menu = OneHotEncoder(handle_unknown='ignore')
ohe_user = OneHotEncoder(handle_unknown='ignore')

menu_df = pd.read_csv(local_menu_csv_path)
user_df = pd.read_csv(local_user_csv_path)

menu_cat_features = menu_df[['diet_labels', 'recipe_name']]
user_cat_features = user_df[['diet_labels', 'preferred_food']]

ohe_menu.fit(menu_cat_features)
ohe_user.fit(user_cat_features)

@app.route('/predictMRS', methods=['POST'])
def predict():
    data = request.json
    
    user_input = data['user_features']
    
    user_cat_features = pd.DataFrame([user_input])
    
    user_features_encoded = ohe_user.transform(user_cat_features).toarray()
    
    # Ensure the user features are repeated for each menu item
    user_features_repeated = np.tile(user_features_encoded, (len(menu_df), 1))
    
    menu_features_encoded = ohe_menu.transform(menu_cat_features).toarray()
    
    # Make predictions for each unique menu item with the same user features
    predictions = model.predict([menu_features_encoded, user_features_repeated])
    
    # Combine the predictions with the menu items
    menu_df['compatibility_score'] = predictions
    
    top_n = 10
    top_recommendations = menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)
    
    # Extract necessary fields for the response
    result = top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']].to_dict(orient='records')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
