from flask import Flask
from controllers.mrs_controller import prediction_bp
# from controllers.ir_controller import ir_controller
import os

os.makedirs('../model', exist_ok=True)
os.makedirs('../csv', exist_ok=True)
# Todo remove ini bang, bad practice
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../serviceKey.json'


app = Flask(__name__)

app.register_blueprint(prediction_bp)
# app.register_blueprint(ir_controller)

# if __name__ == '__main__':
#     # Production
#     # app.run(debug=True, host='0.0.0.0', port=8080)
#     # Development
#     app.run(debug=True, host='127.0.0.1', port=5000)