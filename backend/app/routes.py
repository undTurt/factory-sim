from flask import Blueprint, request, jsonify
from .image_processor import process_grid_image
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@main.route('/process-grid', methods=['POST'])
def process_grid():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        grid_data = process_grid_image(filepath)
        return jsonify(grid_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500