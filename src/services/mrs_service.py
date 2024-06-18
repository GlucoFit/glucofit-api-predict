import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from services.gcs_service import download_file_from_gcs
from config import Config

class MRSService:
    def __init__(self, model):
        print('Log: MRSService initialized')
        self.model = model
        self.ohe_menu = OneHotEncoder(handle_unknown='ignore')
        self.ohe_user = OneHotEncoder(handle_unknown='ignore')
        self.menu_df = None
        self.user_df = None

    def load_and_fit_encoders(self):
        # Download and load menu CSV
        menu_csv_path = self.download_csv(Config.MENU_CSV_PATH_SECRET, 'menu.csv')
        if menu_csv_path:
            self.menu_df = pd.read_csv(menu_csv_path)

        # Download and load user CSV
        user_csv_path = self.download_csv(Config.USER_CSV_PATH_SECRET, 'user.csv')
        if user_csv_path:
            self.user_df = pd.read_csv(user_csv_path)

        # Fit encoders if dataframes are loaded successfully
        if self.menu_df is not None and self.user_df is not None:
            menu_cat_features = self.menu_df[['diet_labels', 'recipe_name']]
            user_cat_features = self.user_df[['diet_labels', 'preferred_food']]
            self.ohe_menu.fit(menu_cat_features)
            self.ohe_user.fit(user_cat_features)
        else:
            raise ValueError("Dataframes not loaded. Check the CSV paths or download process.")

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

        user_cat_features = pd.DataFrame([user_input])
        user_features_encoded = self.ohe_user.transform(user_cat_features).toarray()
        user_features_encoded = np.hstack([user_features_encoded]).astype(np.float32)
        user_features_repeated = np.tile(user_features_encoded, (len(self.menu_df), 1))
        menu_features_encoded = self.ohe_menu.transform(self.menu_df[['diet_labels', 'recipe_name']]).toarray()

        predictions = self.model.predict([menu_features_encoded, user_features_repeated])
        self.menu_df['compatibility_score'] = predictions
        print(predictions)
        top_n = 10
        top_recommendations = self.menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)
        print(top_recommendations)
        return top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']].to_dict(orient='records')
