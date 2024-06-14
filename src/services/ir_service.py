import os
import pandas as pd
import numpy as np
import tensorflow as tf
from services.gcs_service import download_file_from_gcs
from config import Config

class ImageRecognitionService:
    def __init__(self, model, sugar_csv_secret_name):
        print('log calls from IR service Class init')
        self.model = model
        print('log calls from IR service Class - model successfully loaded')
        self.food_names = self.load_food_names(sugar_csv_secret_name)

    def load_food_names(self, sugar_csv_secret_name):
        sugar_csv_path_secret = Config.get_secret(sugar_csv_secret_name)
        csv_bucket_name, csv_blob_name = sugar_csv_path_secret.replace("gs://", "").split("/", 1)

        local_csv_path = './csv/GulaMakanan.csv'

        if not os.path.exists(local_csv_path):
            download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)

        sugar_df = pd.read_csv(local_csv_path, sep=',', header=None, names=['food,sugar(g)'])
        sugar_df['food,sugar(g)'] = sugar_df['food,sugar(g)'].str.strip()
        sugar_df[['food', 'sugar(g)']] = sugar_df['food,sugar(g)'].str.split(',', expand=True)
        sugar_df.drop(columns=['food,sugar(g)'], inplace=True)

        return sugar_df['food'].tolist()

    def preprocess_image(self, image_path):
        img = tf.keras.applications.resnet50.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
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
        return self.food_names[predicted_id - 1]
