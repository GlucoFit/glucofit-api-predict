import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from services.gcs_service import download_file_from_gcs
from config import Config

class ImageRecognitionService:
    def __init__(self, model_path, sugar_csv_secret_name):
        print('Log: ImageRecognitionService initialized')
        self.model = tf.keras.models.load_model(model_path)
        self.food_names, self.sugar_content = self.load_food_data(sugar_csv_secret_name)
        
        # Dummy data to simulate the class indices mapping
        # Replace this with the actual class_indices from your training data
        self.class_names = [
            "apple_pie", "baby_back_ribs", "baklava",
            "beignets", "bibimbap", "breakfast_burrito", "caesar_salad",
            "cannoli", "caprese_salad", "cheesecake", "cheese_plate",
            "chicken_curry",  "chicken_wings", "chocolate_cake", "chocolate_mousse",
            "churros", "club_sandwich",  "creme_brulee",
            "cup_cakes",  "donuts", "dumplings",
            "falafel", "filet_mignon", "fish_and_chips", "french_fries",
            "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread",
            "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "gyoza", "hamburger",
            "hot_dog",  "ice_cream", "lasagna",
            "macaroni_and_cheese", "macarons", "miso_soup",
            "nachos", "omelette", "onion_rings", "pancakes",
            "pizza", "prime_rib",
            "ramen", "red_velvet_cake", "samosa", "sashimi",
            "spaghetti_bolognese", "spaghetti_carbonara",
            "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
            "waffles"
        ]  # Example
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
        img = tf.keras.preprocessing.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array * (1.0 / 255.0)  # Scale the image array
        return img_array

    def predict_image(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_id = self.idx_to_class[predicted_class]
        confidence = np.max(predictions)
        return predicted_id, confidence

    def get_food_name_by_id(self, predicted_id):
        return self.food_names[predicted_id]

    def get_sugar_content_by_id(self, predicted_id):
        return self.sugar_content[predicted_id]

    def display_prediction(self, image_path):
        predicted_id, confidence = self.predict_image(image_path)
        predicted_class = self.idx_to_class[predicted_id]

        print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

        # Display the image
        plt.imshow(tf.keras.preprocessing.load_img(image_path))
        plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
        plt.axis('off')
        plt.show()
