import os
from google.cloud import secretmanager

class Config:
    # GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

    @staticmethod
    def get_secret(secret_id):
        client = secretmanager.SecretManagerServiceClient()
        # project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        project_id = "capstone-playground-423804"
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(name=name)
        return response.payload.data.decode('UTF-8')

    MODEL_PATH_SECRET = get_secret('DEEP_LEARNING_CBF_MODEL_PATH')
    MODEL2_PATH_SECRET = get_secret('RESNET50_MODEL_PATH')
    MENU_CSV_PATH_SECRET = get_secret('MENU_CSV_PATH')
    USER_CSV_PATH_SECRET = get_secret('USER_CSV_PATH')
    SUGAR_CSV_PATH_SECRET = get_secret('SUGAR_CSV_PATH')
