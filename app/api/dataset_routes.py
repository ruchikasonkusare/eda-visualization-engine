from flask import Blueprint, request, jsonify
from app.utils.validators import allowed_file
from app.services.dataset_service import save_dataset,get_dataset_summary,load_dataset_by_id,save_cleaned_dataset
from app.services.cleaning_service import run_cleaning_pipeline,basic_structure_check
from app.services.visualization_service import recommendation_visulaizations
from app.services.plot_generation_service import generate_plot
from app.services.report_generate_service import generate_pdf_report

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
        
@dataset_bp.route("/<dataset_id>/visualizations",methods=["GET"])
def get_visualizations(dataset_id):
    try:
        df,_,_ =load_dataset_by_id(dataset_id)
        recommendation=recommendation_visulaizations(df)
        
        return jsonify({
            "status":'success',
            "message":"Visualization recommendation generated",
            "data":recommendation
        }),200
        
    except Exception as e:
        return jsonify({
            "status":"error",
            "message":str(e)
        }),500
        
        
@dataset_bp.route("/<dataset_id>/report",methods=["GET"])
def create_dataset_report(dataset_id):
    try:
        df,_,_=load_dataset_by_id(dataset_id)
        
        summary=get_dataset_summary(dataset_id)
        recommendations=recommendation_visulaizations(df)
        visual=[]
        
        for rec in recommendations:
            for chart in rec.get('recommended_charts'):
                try:
                    plot=generate_plot(
                        df,
                        chart['chart'],
                        rec['columns']
                    )
                    plot["insight"]=chart.get("insight",'')
                    visual.append(plot)
                
                except Exception as e:
                    print("Plot failed",e)
                    continue
                
        print("TOTAL VISUALS:", len(visual))
                
        report_url=generate_pdf_report(
            dataset_id,
            summary,
            visual
        )
        
        return jsonify({
            "status":"success",
            "message":"PDF report generated successfully",
            "report_url":report_url
        }),200
    
    except Exception as e:
        return jsonify({
            "status":"success",
            "message":str(e)
        }),500