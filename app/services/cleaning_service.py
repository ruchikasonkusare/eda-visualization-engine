import os
import pandas as pd


def basic_structure_check(df:pd.DataFrame)->dict:
    return {
        'rows':int(df.shape[0]),
        'columns':int(df.shape[1]),
        'column_names':df.columns.tolist(),
        'data_types':df.dtypes.astype(str).to_dict(),
        'missing_values':df.isnull().sum().to_dict(),
        "unique_values":df.nunique().to_dict()
    }