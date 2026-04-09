import os
import pandas as pd
import uuid
from app.services.cleaning_service import basic_structure_check

upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)

def save_dataset(file):
    ext=file.filename.rsplit('.',1)[1].lower()
    dataset_id=str(uuid.uuid4())
    filename=f"{dataset_id}.{ext}"
    file_path=os.path.join(upload_folder,filename)
    file.save(file_path)
    
    if ext=='csv':
        df=pd.read_csv(file_path)
    elif ext=='json':
        df=pd.read_json(file_path)
    elif ext=='xlsx':
        df=pd.read_excel(file_path)
    
    return {
        'dataset_id': dataset_id,
        'file_path': file_path,
        'shape': df.shape,
    }
    
def get_dataset_summary(dataset_id):
    
    df,file_path,extension = load_dataset_by_id(dataset_id)
    
    summary=basic_structure_check(df)
    
    summary['duplicated_rows']=int(df.duplicated().sum())
    summary["preview"]=df.head(5).to_dict(orient='records')
    
    return summary
    
def load_dataset_by_id(dataset_id):
    file_path=None
    
    for file in os.listdir(upload_folder):
        if file.startswith(dataset_id):
            file_path=os.path.join(upload_folder,file)
            break
    
    if not file_path:
        raise FileNotFoundError("Dataset not found")
    
    extension=file_path.rsplit(".",1)[1].lower()
    
    if extension == 'csv':
        df=pd.read_csv(file_path)
    
    elif extension =='json':
        df=pd.read_json(file_path)
        
    elif extension =='xlsx':
        df=pd.read_excel(file_path,engine='openpyxl')
    
    else:
        raise ValueError("Unsupported file format")

    return df,file_path,extension