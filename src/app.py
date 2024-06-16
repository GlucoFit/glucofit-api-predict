import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
# from controllers.mrs_controller import prediction_bp
from controllers.ir_controller import ir_controller
from google.oauth2 import service_account

os.makedirs('./model', exist_ok=True)
os.makedirs('./csv', exist_ok=True)
# Todo remove ini bang, bad practice

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(
    "./serviceKey.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

app = Flask(__name__)

# app.register_blueprint(prediction_bp)
app.register_blueprint(ir_controller)

# if __name__ == '__main__':
# Production
#	from waitress import serve
#	serve(app, host="0.0.0.0", port=5000)
#     app.run(debug=True, host='0.0.0.0', port=8080)
#     # Development
#     app.run(debug=True, host='127.0.0.1', port=5000)
