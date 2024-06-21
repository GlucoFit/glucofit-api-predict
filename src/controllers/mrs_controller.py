from flask import Blueprint, request, jsonify
from services.mrs_service import MRSService
from models import model1  # Assuming model1 is imported correctly
mrs_controller = Blueprint('mrs_controller', __name__)

@mrs_controller.route('/predictMRS', methods=['POST'])
def predict():
    data = request.json
    user_input = data['user_features']
    
    # Assuming CSV loading is no longer necessary here
    # menu_df = load_csv('./csv/menu.csv')
    # user_df = load_csv('./csv/user.csv')
    
    model_service = MRSService(model1)
    model_service.load_and_fit_encoders()  # This will download and load CSVs internally
    
    try:
        result = model_service.predict(user_input)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 500