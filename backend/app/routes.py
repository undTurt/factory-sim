from flask import Blueprint, request, jsonify
from .image_processor import process_grid_image, decode_base64_image
import os
import cv2

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@main.route('/process-grid', methods=['POST'])
def process_grid():
    try:
        # Get grid division parameters
        num_cols = int(request.args.get('cols', 4))
        num_rows = int(request.args.get('rows', 6))
        
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        else:
            # Handle base64 image data
            json_data = request.get_json()
            if not json_data or 'image' not in json_data:
                return jsonify({'error': 'No image data provided'}), 400
            
            image = decode_base64_image(json_data['image'])
            filepath = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
            cv2.imwrite(filepath, image)

        # Process the image with grid divisions
        result = process_grid_image(filepath, num_cols, num_rows)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500