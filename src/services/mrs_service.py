import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from services.gcs_service import download_file_from_gcs
from config import Config

class MRSService:
    def __init__(self, model):
        self.model = model
        self.ohe_menu = OneHotEncoder(handle_unknown='ignore')
        self.ohe_user = OneHotEncoder(handle_unknown='ignore')
        self.menu_df = None
        self.user_df = None

    def load_and_fit_encoders(self):
        # Download and load menu CSV
        menu_csv_path = self.download_csv(Config.MENU_CSV_PATH_SECRET)
        self.menu_df = pd.read_csv(menu_csv_path)

        # Download and load user CSV
        user_csv_path = self.download_csv(Config.USER_CSV_PATH_SECRET)
        self.user_df = pd.read_csv(user_csv_path)

        # Fit encoders
        menu_cat_features = self.menu_df[['diet_labels', 'recipe_name']]
        user_cat_features = self.user_df[['diet_labels', 'preferred_food']]
        self.ohe_menu.fit(menu_cat_features)
        self.ohe_user.fit(user_cat_features)
    
    def download_csv(self, csv_secret_path):
        csv_path_secret = Config.get_secret(csv_secret_path)
        csv_bucket_name, csv_blob_name = csv_path_secret.replace("gs://", "").split("/", 1)

        local_csv_path = f'./csv/{os.path.basename(csv_blob_name)}'

        if not os.path.exists(local_csv_path):
            download_file_from_gcs(csv_bucket_name, csv_blob_name, local_csv_path)

        return local_csv_path
    
    def predict(self, user_input):
        if self.menu_df is None or self.user_df is None:
            raise ValueError("Dataframes not loaded. Call load_and_fit_encoders first.")

        user_cat_features = pd.DataFrame([user_input])
        user_features_encoded = self.ohe_user.transform(user_cat_features).toarray()
        user_features_repeated = np.tile(user_features_encoded, (len(self.menu_df), 1))
        menu_features_encoded = self.ohe_menu.transform(self.menu_df[['diet_labels', 'recipe_name']]).toarray()
        predictions = self.model.predict([menu_features_encoded, user_features_repeated])
        self.menu_df['compatibility_score'] = predictions
        top_n = 10
        top_recommendations = self.menu_df.sort_values(by='compatibility_score', ascending=False).head(top_n)
        return top_recommendations[['recipe_name', 'diet_labels', 'compatibility_score']].to_dict(orient='records')
