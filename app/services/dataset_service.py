import os
import pandas as pd
import uuid

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
    file_path=None
    for file in os.listdir(upload_folder):
        if file.startswith(dataset_id):
            file_path=os.path.join(upload_folder,file)
            break
        
    if not file_path:
        raise FileNotFoundError("Dataset not found")
    
    if file_path.endswith('.csv'):
        df=pd.read_csv(file_path)
    
    elif file_path.endswith('.json'):
        df=pd.read_json(file_path)
    
    elif file_path.endswith('.xlsx'):
        df=pd.read_excel(file_path)
        
    else:
        raise ValueError('Unsupported file format')
    
    return {
        'shape':df.shape,
        'columns':df.columns.tolist(),
        'dtypes':df.dtypes.astype(str).to_dict(),
        'missing_values':df.isnull().sum().to_dict(),
        'duplicate_rows':int(df.duplicated().sum()),
        'preview':df.head(5).to_dict(orient='records')
    }