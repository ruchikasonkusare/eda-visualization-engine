from flask import Blueprint, request, jsonify

dataset_bp = Blueprint('dataset_bp', __name__)

@dataset_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'EDA API is running smoothly'
    }), 200
    
    