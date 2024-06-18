import os
import pandas as pd
import numpy as np
import tensorflow as tf
from services.gcs_service import download_file_from_gcs
from config import Config

class ImageRecognitionService:
    def __init__(self, model, sugar_csv_secret_name):
        print('Log: ImageRecognitionService initialized')
        self.model = model
        self.food_names, self.sugar_content = self.load_food_data(sugar_csv_secret_name)

    def load_food_data(self, sugar_csv_secret_name):
        sugar_csv_path_secret = Config.get_secret(sugar_csv_secret_name)
        csv_bucket_name, csv_blob_name = sugar_csv_path_secret.replace("gs://", "").split("/", 1)

        local_csv_path = './csv/GulaMakanan.csv'

        def download_csv():
            print(f"Downloading CSV from GCS: {csv_bucket_name}/{csv_blob_name}")
            download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)

        # Check if the CSV file already exists locally
        if not os.path.exists(local_csv_path):
            download_csv()

        # Check if the file exists after download or before loading
        if os.path.exists(local_csv_path):
            print(f"CSV file exists: {local_csv_path}")
        else:
            print(f"CSV file does not exist: {local_csv_path}")
            return None, None

        # Try loading the CSV
        try:
            sugar_df = pd.read_csv(local_csv_path, sep=',', header=None, names=['food,sugar(g)'])
            sugar_df['food,sugar(g)'] = sugar_df['food,sugar(g)'].str.strip()
            sugar_df[['food', 'sugar(g)']] = sugar_df['food,sugar(g)'].str.split(',', expand=True)
            sugar_df.drop(columns=['food,sugar(g)'], inplace=True)
            print("CSV loaded successfully")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error loading CSV: {e}")
            print("Attempting to re-download the CSV...")
            download_csv()
            
            # Try loading the CSV again after re-download
            try:
                sugar_df = pd.read_csv(local_csv_path, sep=',', header=None, names=['food,sugar(g)'])
                sugar_df['food,sugar(g)'] = sugar_df['food,sugar(g)'].str.strip()
                sugar_df[['food', 'sugar(g)']] = sugar_df['food,sugar(g)'].str.split(',', expand=True)
                sugar_df.drop(columns=['food,sugar(g)'], inplace=True)
                print("CSV loaded successfully after re-download")
            except Exception as e:
                print(f"Error loading CSV after re-download: {e}")
                return None, None

        food_names = sugar_df['food'].tolist()
        sugar_content = sugar_df['sugar(g)'].tolist()

        return food_names, sugar_content

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        return img_array

    def predict_image(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_id = predicted_class + 1
        return predicted_id

    def get_food_name_by_id(self, predicted_id):
        return self.food_names[predicted_id]

    def get_sugar_content_by_id(self, predicted_id):
        return self.sugar_content[predicted_id]
