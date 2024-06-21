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
        self.sugar_df = self.load_food_data(sugar_csv_secret_name)
        
        self.class_names = [
            "apple_pie", "baby_back_ribs", "baklava",
            "beignets", "bibimbap", "breakfast_burrito", "caesar_salad",
            "cannoli", "caprese_salad", "cheesecake", "cheese_plate",
            "chicken_curry", "chicken_wings", "chocolate_cake", "chocolate_mousse",
            "churros", "club_sandwich", "creme_brulee",
            "cup_cakes", "donuts", "dumplings",
            "falafel", "filet_mignon", "fish_and_chips", "french_fries",
            "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread",
            "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "gyoza", "hamburger",
            "hot_dog", "ice_cream", "lasagna",
            "macaroni_and_cheese", "macarons", "miso_soup",
            "nachos", "omelette", "onion_rings", "pancakes",
            "pizza", "prime_rib",
            "ramen", "red_velvet_cake", "samosa", "sashimi",
            "spaghetti_bolognese", "spaghetti_carbonara",
            "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
            "waffles"
        ]
        self.idx_to_class = {i: class_name for i, class_name in enumerate(self.class_names)}

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

        # Try loading the CSV
        try:
            sugar_df = pd.read_csv(local_csv_path, sep=',', header=0, names=['food','sugar(g)'])
            print("CSV loaded successfully")
            print(sugar_df.head())
            return sugar_df
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error loading CSV: {e}")
            return None

    def preprocess_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array * (1.0 / 255.0)  # Scale the image array
        return img_array

    def predict_image(self, image_path):
        print(f"Log: Preprocessing image: {image_path}")
        img_array = self.preprocess_image(image_path)
        print(f"Log: Image preprocessed. Shape: {img_array.shape}")

        predictions = self.model.predict(img_array)
        print(f"Log: Predictions: {predictions}")

        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        print(f"Log: Predicted class index: {predicted_class}, Confidence: {confidence}")
        return predicted_class, confidence

    def get_food_name_by_id(self, predicted_id):
        print(f"Log: Getting food name for ID: {predicted_id}")
        return self.class_names[predicted_id]

    def get_sugar_content_by_id(self, predicted_id):
        if self.sugar_df is None:
            print("Log: Sugar data is not available")
            return "No data available"

        try:
            print(f"Log: Getting sugar content for ID: {predicted_id}")
            food_name = self.class_names[predicted_id]
            # print(type(food_name))
            # print(food_name)
            normalized_food_name = food_name.replace('_', ' ').lower()
            sugar_content = self.sugar_df[self.sugar_df['food'].str.lower().str.replace('_', ' ') == normalized_food_name]['sugar(g)'].values
            if sugar_content.size > 0:
                return sugar_content[0]
            else:
                return "No data available"
        except IndexError:
            return "No data available"