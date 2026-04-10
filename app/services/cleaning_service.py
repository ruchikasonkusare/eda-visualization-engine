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

# ================IDENTIFIERS===========================
def is_identifier_column(series: pd.Series, col_name: str) -> bool:
    col_lower = col_name.lower()

    identifier_keywords = [
        "id", "customerid", "customer_id",
        "userid", "user_id", "rowno",
        "row_no", "index", "serial"
    ]

    # keyword based detection
    if any(keyword in col_lower for keyword in identifier_keywords):
        return True

    # high uniqueness means likely identifier
    uniqueness_ratio = series.nunique(dropna=True) / max(len(series), 1)

    if uniqueness_ratio > 0.95:
        return True

    return False
    
# =============DATEATIME FORMAT NORMALIZATION===========
def datetime_check(df:pd.DataFrame)->pd.DataFrame:
    new_df=df.copy()
    
    for col in new_df.select_dtypes(include=['object']).columns:
        sample=new_df[col].dropna().astype(str).head(10)
        
        if sample.empty:
            continue
        success_ratio=0
        parsed=pd.to_datetime(sample,errors='coerce')
        
        success_ratio=parsed.notna().mean()
        
        if success_ratio>0.8:
            new_df[col]=pd.to_datetime(new_df[col],errors='coerce')
    return new_df

# =============ORDINAL DETECTION=================

COMMON_ORDERS = [
    ["low", "medium", "high"],
    ["poor", "average", "good", "excellent"],
    ["bronze", "silver", "gold"],
    ["s", "m", "l", "xl"],
    ["freshman", "sophomore", "junior", "senior"]
]

def detect_ordinal(df):
    ordinal_column={}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = set(
            df[col].dropna().astype(str).str.lower().unique())
        
        for order in COMMON_ORDERS:
            if unique_values.issubset(set(order)):
                ordinal_column[col] = order
                print(f"Column '{col}' detected as ordinal with order: {order}")
                break
    return ordinal_column

# =================================COLUMN TYPE DETECTION=========

def col_types(df:pd.DataFrame)->list:
    
    ordinal_info = detect_ordinal(df)
    result=[]
    for col in df.columns:
        dtype=str(df[col].dtype)
        col_type="text"
        
        
        if is_identifier_column(df[col],col):
            col_type="identifier"
            
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_type="numeric"
            
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type="datetime"
            
        elif col in ordinal_info:
            col_type="ordinal"
            
        elif df[col].nunique() < 20:
            col_type="categorical"
            
        elif df[col].nunique(dropna=True) == 2:
            col_type="binary"
            
        result.append({
            'Column': col,
            'Data_Type': dtype,
            'Inferrd_Type': col_type,
            'Unique_Values': df[col].nunique(dropna=True),
            'Ordinal_Order': ordinal_info.get(col, None)
        })
        
    return result

# ====================NULL SUMMARY================

def null_check(df:pd.DataFrame)->dict:
    return df.isnull().sum().to_dict()

# ===================NULL HANDLER======================

def smart_null_handler(df: pd.DataFrame):
    new_df = df.copy()
    report = {}

    identity_cols = {
        "first name", "last name", "email",
        "phone number", "address",
        "position", "company"
    }

    for col in list(new_df.columns):
        missing_pct = new_df[col].isna().mean() * 100

        if missing_pct == 0:
            continue

        action = ""

        if missing_pct < 5:
            new_df = new_df[new_df[col].notna()]
            action = "removed_rows"

        elif missing_pct > 60:
            new_df = new_df.drop(columns=[col])
            action = "dropped_column"

        elif pd.api.types.is_numeric_dtype(new_df[col]):
            skew = abs(new_df[col].dropna().skew())

            if skew < 1:
                new_df[col] = new_df[col].fillna(new_df[col].mean())
                action = "filled_mean"
            else:
                new_df[col] = new_df[col].fillna(new_df[col].median())
                action = "filled_median"

        elif pd.api.types.is_datetime64_any_dtype(new_df[col]):
            new_df[col] = new_df[col].ffill()
            action = "forward_fill"

        else:
            if col.lower() in identity_cols:
                new_df[col] = new_df[col].fillna("Unknown")
                action = "filled_unknown"
            else:
                mode_value = (
                    new_df[col].mode().iloc[0]
                    if not new_df[col].mode().empty
                    else "Unknown"
                )
                new_df[col] = new_df[col].fillna(mode_value)
                action = "filled_mode"

        report[col] = {
            "missing_percent": round(missing_pct, 2),
            "action": action
        }

    return new_df, report

# ==========================DUPLICATE HANDLING============

def duplicate_check(df:pd.DataFrame)->int:
    return int(df.duplicated().sum())

def remove_duplicates(df:pd.DataFrame):
    before=len(df)
    new_df=df.drop_duplicates().copy()
    
    return new_df,{
        "before":before,
        "after":len(new_df),
        "removed":before-len(new_df)
    }


# ==========================OUTLIER BOUNDS=================
def get_iqr_bounds(df: pd.DataFrame, threshold: float = 1.5) -> dict:
    bounds = {}

    for col in df.select_dtypes(include=["number"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        bounds[col] = (
            q1 - threshold * iqr,
            q3 + threshold * iqr
        )

    return bounds

# ======================OUTLIER DETECT==========================

def detect_outliers(df: pd.DataFrame, threshold: float = 1.5) -> dict:
    bounds = get_iqr_bounds(df, threshold)
    outlier_report = {}

    for col, (lower, upper) in bounds.items():
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_report[col] = int(mask.sum())

    return outlier_report

# ===================OUTLIER TREATMENT==========================


def treat_outliers(df: pd.DataFrame, threshold: float = 1.5):
    new_df = df.copy()
    bounds = get_iqr_bounds(new_df, threshold)

    report = {}

    for col, (lower, upper) in bounds.items():
        before = ((new_df[col] < lower) | (new_df[col] > upper)).sum()

        new_df[col] = new_df[col].clip(lower=lower, upper=upper)

        after = ((new_df[col] < lower) | (new_df[col] > upper)).sum()

        report[col] = {
            "before": int(before),
            "after": int(after),
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }

    return new_df, report


# ========================MASTER CLEANING PIPELINE=========

def run_cleaning_pipeline(df:pd.DataFrame):
    report={}
    
    cleaned_df=df.copy()
    
    report['structure_before']=basic_structure_check(cleaned_df)
    
    cleaned_df=datetime_check(cleaned_df)
    
    report["column_types"]=col_types(cleaned_df)
    
    cleaned_df,null_report=smart_null_handler(cleaned_df)
    report["missing_value_treatment"]=null_report
    
    cleaned_df,dupli_report=remove_duplicates(cleaned_df)
    report["duplicates"]=dupli_report

    report["outliers_before"]=detect_outliers(cleaned_df)
    
    cleaned_df,outlier_report=treat_outliers(cleaned_df)
    report["outliers_treatment"]=outlier_report
    
    report["structure_after"]=basic_structure_check(cleaned_df)
    
    return cleaned_df,report
