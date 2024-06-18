import os
import tensorflow as tf
from services.gcs_service import download_file_from_gcs
from config import Config

def load_model1():
    local_model_path = './model/model1.h5'
    model_path_secret = Config.MODEL_PATH_SECRET
    model_bucket_name, model_blob_name = model_path_secret.replace("gs://", "").split("/", 1)
    
    # Check if the model file already exists locally
    if not os.path.exists(local_model_path):
        download_file_from_gcs(model_bucket_name, model_blob_name, local_model_path)
    
    return tf.keras.models.load_model(local_model_path)