from flask import Blueprint, request, jsonify
from services.ir_service import ImageRecognitionService
from models import model2
from repositories.csv_repository import load_csv

ir_controller = Blueprint('ir_controller', __name__)

# Load GulaMakanan.csv for food names and sugar content
sugar_csv_secret_name = 'SUGAR_CSV_PATH'  # Replace with your secret name

@ir_controller.route('/predictIR', methods=['POST'])
def predict_ir():
    # Assuming you are receiving the image file in the request
    image_file = request.files['image']

    image_recognition_service = ImageRecognitionService(model2, sugar_csv_secret_name)
    # # Save the image to a temporary location (you may adjust this based on your application)
    # image_path = './temp/' + image_file.filename
    # image_file.save(image_path)

    # # Perform prediction using the ImageRecognitionService
    # predicted_id = image_recognition_service.predict_image(image_path)
    # food_name = image_recognition_service.get_food_name_by_id(predicted_id)

    # # Optional: Get sugar content based on predicted_id
    # sugar_content = image_recognition_service.get_sugar_content_by_id(predicted_id)

    # # Return prediction result as JSON
    # result = {
    #     'food_name': food_name,
    #     'sugar_content': sugar_content  # You can adjust this based on your service method
    # }

    # return jsonify(result)
    return jsonify({"message": "success on loading"})