import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from services.gcs_service import download_file_from_gcs
from config import Config

class ImageRecognitionService:
    def __init__(self, model, sugar_csv_secret_name):
        print('Log calls from IR service Class init')
        self.model = model
        print('Log calls from IR service Class - model successfully loaded')
        self.food_names = self.load_food_names(sugar_csv_secret_name)
        self.sugar_df = self.load_sugar_data(sugar_csv_secret_name)

    def load_food_names(self, sugar_csv_secret_name):
        # Define food names directly here
        return [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
            "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
            "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate",
            "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
            "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
            "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
            "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
            "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
            "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
            "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
            "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
            "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
            "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
            "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
            "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
            "tuna_tartare", "waffles"
        ]

    def load_sugar_data(self, sugar_csv_secret_name):
        sugar_csv_path_secret = Config.get_secret(sugar_csv_secret_name)
        csv_bucket_name, csv_blob_name = sugar_csv_path_secret.replace("gs://", "").split("/", 1)

        local_csv_path = './csv/GulaMakanan.csv'

        if not os.path.exists(local_csv_path):
            download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)

        sugar_df = pd.read_csv(local_csv_path, sep=',', header=None, names=['food,sugar(g)'])
        sugar_df['food,sugar(g)'] = sugar_df['food,sugar(g)'].str.strip()
        sugar_df[['food', 'sugar(g)']] = sugar_df['food,sugar(g)'].str.split(',', expand=True)
        sugar_df.drop(columns=['food,sugar(g)'], inplace=True)

        return sugar_df

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(224, 224))  # Load image and resize
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image for ResNet50
        return img_array

    def predict_image(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array)  # Make prediction
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get the class index with the highest probability
        predicted_id = predicted_class + 1  # Convert 0-based index to 1-based ID
        return predicted_id

    def get_food_name_by_id(self, predicted_id):
        return self.food_names[predicted_id - 1]

    def get_sugar_content_by_id(self, predicted_id):
        predicted_label = self.get_food_name_by_id(predicted_id)
        try:
            sugar_content = self.sugar_df.loc[self.sugar_df['food'] == predicted_label, 'sugar(g)'].values[0]
            return sugar_content
        except IndexError:
            return None

