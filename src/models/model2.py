import os
import tensorflow as tf
from services.gcs_service import download_file_from_gcs
from config import Config

def load_model2():
    local_model_path = './model/model2.h5'
    model_path_secret = Config.MODEL2_PATH_SECRET
    model_bucket_name, model_blob_name = model_path_secret.replace("gs://", "").split("/", 1)
    
    # Check if the model file already exists locally
    if not os.path.exists(local_model_path):
        print(f"Downloading model from GCS: {model_bucket_name}/{model_blob_name}")
        download_file_from_gcs(model_bucket_name, model_blob_name, local_model_path)
    
    # Check if the file exists after download
    if os.path.exists(local_model_path):
        print(f"Model file exists: {local_model_path}")
    else:
        print(f"Model file does not exist: {local_model_path}")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(local_model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

