from flask import Blueprint, request, jsonify
from app.utils.validators import allowed_file
from app.services.dataset_service import save_dataset,get_dataset_summary,load_dataset_by_id,save_cleaned_dataset
from app.services.cleaning_service import run_cleaning_pipeline,basic_structure_check

dataset_bp = Blueprint('datasets', __name__)

@dataset_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'EDA API is running smoothly'
    }), 200
    
@dataset_bp.route('', methods=['POST'])
def upload_dataset():
    print('CONTENT-TYPE:', request.content_type)
    print('FILES:', request.files)
    print('FORM:', request.form)
    if 'files' not in request.files:
        return jsonify({
            'status':'error',
            'message':'No file uploaded'
        }),400
    
    file=request.files['files']
    
    if file.filename=='':
        return jsonify({
            'status':'error',
            'message':'Empty filename'
        }),400
        
    if not allowed_file(file.filename):
        return jsonify({
            'status':'error',
            'message':'Unsupported file type'
        }),400
    
    result=save_dataset(file)
    
    return jsonify({
        'status':'success',
        'message':'File uploaded successfully',
        'data':result
    }),201
    
    
@dataset_bp.route('/<dataset_id>/summary',methods=["GET"])
def dataset_summary(dataset_id):
    try:
        summary=get_dataset_summary(dataset_id)
        
        return jsonify({
            'status':'success',
            'message':'Dataset summary fetched successfully.',
            'data':summary
        }),200
        
    except FileNotFoundError as e:
        return jsonify({
            'status':'error',
            'message':str(e)
        }),404
        
    except Exception as e:
        return jsonify({
            'status':'error',
            'message':str(e)
        }),500

@dataset_bp.route("/<dataset_id>/cleaning", methods=["PATCH"])
def clean_dataset(dataset_id):
    try:
        df, file_path, extension = load_dataset_by_id(dataset_id)

        cleaned_df, report = run_cleaning_pipeline(df)

        save_cleaned_dataset(cleaned_df, file_path, extension)

        return jsonify({
            "status": "success",
            "message": "Detailed cleaning pipeline applied",
            "data": report
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500