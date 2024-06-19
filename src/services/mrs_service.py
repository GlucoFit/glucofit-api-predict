import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from services.gcs_service import download_file_from_gcs
from config import Config
import pickle
import joblib

class MRSService:
    def __init__(self, model):
        print('Log: MRSService initialized')
        self.model = model
        self.ohe_menu = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.ohe_user = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.menu_df = None
        self.user_df = None
        self.menu_features_encoded = None
        self.user_features_encoded = None
        self.menu_scaler = StandardScaler()
        self.user_scaler = StandardScaler()

    def load_and_fit_encoders(self):
        menu_csv_path = self.download_csv(Config.MENU_CSV_PATH_SECRET, 'menu.csv')
        if menu_csv_path:
            self.menu_df = pd.read_csv(menu_csv_path)

        user_csv_path = self.download_csv(Config.USER_CSV_PATH_SECRET, 'user.csv')
        if user_csv_path:
            self.user_df = pd.read_csv(user_csv_path)
            self.user_df = pd.concat([self.user_df]* int(np.ceil(len(self.menu_df) / len(self.user_df))), ignore_index=True)
            self.user_df = self.user_df.iloc[:len(self.menu_df)]

        if self.menu_df is not None and self.user_df is not None:
            menu_cat_features = self.menu_df[['diet_labels', 'recipe_name']]
            user_cat_features = self.user_df[['diet_labels', 'preferred_food']]
            self.menu_features_encoded = self.ohe_menu.fit_transform(menu_cat_features)
            self.user_features_encoded = self.ohe_user.fit_transform(user_cat_features)
            self.menu_features_encoded = np.hstack([self.menu_features_encoded]).astype(np.float32)
            self.user_features_encoded = np.hstack([self.user_features_encoded]).astype(np.float32)

            self.menu_features_encoded = self.menu_scaler.fit_transform(self.menu_features_encoded)
            self.user_features_encoded = self.user_scaler.fit_transform(self.user_features_encoded)
        else:
            raise ValueError("Dataframes not loaded. Check the CSV paths or download process.")

        # Print shapes for debugging
        print(f"Menu features encoded shape: {self.menu_features_encoded.shape}")
        print(f"User features encoded shape: {self.user_features_encoded.shape}")

    def download_csv(self, csv_secret_path, local_csv_name):
        print(f"Log: Retrieving secret for path: {csv_secret_path}")
        csv_bucket_name, csv_blob_name = csv_secret_path.replace("gs://", "").split("/", 1)
        local_csv_path = f'./csv/{local_csv_name}'

        def download_csv():
            print(f"Downloading CSV from GCS: {csv_bucket_name}/{csv_blob_name}")
            download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)

        if not os.path.exists(local_csv_path):
            download_csv()

        if os.path.exists(local_csv_path):
            print(f"CSV file exists: {local_csv_path}")
        else:
            print(f"CSV file does not exist: {local_csv_path}")
            return None

        try:
            pd.read_csv(local_csv_path)
            print("CSV loaded successfully")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error loading CSV: {e}")
            print("Attempting to re-download the CSV...")
            download_csv()
            try:
                pd.read_csv(local_csv_path)
                print("CSV loaded successfully after re-download")
            except Exception as e:
                print(f"Error loading CSV after re-download: {e}")
                return None

        return local_csv_path
    
    def predict(self, user_input):
        if self.menu_df is None or self.user_df is None:
            raise ValueError("Dataframes not loaded. Call load_and_fit_encoders first.")

        new_user_cat_features = pd.DataFrame([user_input])
        new_user_cat_encoded = self.ohe_user.transform(new_user_cat_features)
        new_user_features_encoded = np.hstack([new_user_cat_encoded]).astype(np.float32)

        # Scale the new user input features
        new_user_features_scaled = self.user_scaler.transform(new_user_features_encoded)

        # Repeat new user input features to match the length of menu features
        new_user_features_tiled = np.tile(new_user_features_scaled, (len(self.menu_features_encoded), 1))

        # Print shapes for debugging
        print(f"New user features encoded shape: {new_user_features_encoded.shape}")
        print(f"New user features tiled shape: {new_user_features_tiled.shape}")

        # Ensure the shapes match the model's expected input
        if new_user_features_tiled.shape[1] != self.menu_features_encoded.shape[1]:
            raise ValueError(f"Shape mismatch: Model expects {self.menu_features_encoded.shape[1]} features, but got {new_user_features_tiled.shape[1]}")

        predictions = self.model.predict([self.menu_features_encoded, new_user_features_tiled])
        self.menu_df['compatibility_score'] = predictions
        print(predictions)
        top_n = 10
        top_recommendations = self.menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)
        print(top_recommendations)
        return top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']].to_dict(orient='records')
