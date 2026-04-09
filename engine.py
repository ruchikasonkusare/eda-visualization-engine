import pandas as pd
import os


# ============================================LOAD FILE=========================================================
def load_file(file_path=None):
    if file_path is None:
        files=os.listdir('data/')
        if len(files) > 0:
            file_path = os.path.join('data/', files[0])
        else:
            print("No files found in the directory.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.csv'):
        df=pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df=pd.read_excel(file_path)
    else:        
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file."
    )
    return df,file_path


# ===========================================BASIC STRUCTURE CHECK=========================================================
def basic_structure_check(df):
    structure_summary = {
        'Number of Rows': df.shape[0],
        'Number of Columns': df.shape[1],
        'Column Names': df.columns.tolist(),
        'Data Types': df.dtypes.to_dict(),
        'Missing Values': df.isnull().sum().to_dict(),
        'Unique Values': df.nunique().to_dict()
    }
    return structure_summary

# ==========================================DATETIME CHECK=========================================================
def datetime_check(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Column '{col}' converted to datetime.")
            except (ValueError, TypeError):
                print(f"Column '{col}' cannot be converted to datetime.")
    return df


# ===========================================ORDINAL CHECK=========================================================

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
        unique_values = set(df[col].dropna().astype(str).str.lower().unique())
        for order in COMMON_ORDERS:
            if unique_values.issubset(set(order)):
                ordinal_column[col] = order
                print(f"Column '{col}' detected as ordinal with order: {order}")
                break
    return ordinal_column


# ==========================================COLUMN TYPES CHECK=========================================================

def col_types(df):
    
    ordinal_info = detect_ordinal(df)
    rows=[]
    for col in df.columns:
        dtype=str(df[col].dtype)
        col_type="text"
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type="numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type="datetime"
        elif col in ordinal_info:
            col_type="ordinal"
        elif df[col].nunique() < 20:
            col_type="categorical"
        rows.append({
            'Column': col,
            'Data Type': dtype,
            'Inferred Type': col_type,
            'Unique Values': df[col].nunique(),
            'Ordinal Order': ordinal_info.get(col, None)
        })
    return pd.DataFrame(rows)


# ===========================================NULL CHECK=========================================================
def null_check(df):
    null_summary = df.isna().sum().reset_index()
    null_summary.columns = ['Column', 'Null Count']
    return null_summary

def smart_null_handler(df):
    new_df = df.copy()

    for col in new_df.columns:
        missing_pct = new_df[col].isna().mean() * 100

        if missing_pct == 0:
            continue

        print(f"\n{col}: {missing_pct:.2f}% missing")
        # print(new_df)

        if missing_pct < 5:
            print(f"{col}→ Removing rows")
            new_df = new_df[new_df[col].notna()]

        elif missing_pct > 60:
            print(f"{col}→ Dropping column")
            new_df = new_df.drop(columns=[col])

        elif pd.api.types.is_numeric_dtype(new_df[col]):
            skew = abs(new_df[col].dropna().skew())

            if skew < 1:
                print(f"{col}→ Filling with mean")
                new_df[col] = new_df[col].fillna(new_df[col].mean())
            else:
                print(f"{col}→ Filling with median")
                new_df[col] = new_df[col].fillna(new_df[col].median())

        elif pd.api.types.is_datetime64_any_dtype(new_df[col]):
            print(f"{col}→ Forward fill")
            new_df[col] = new_df[col].ffill()

        else:
            identity_col=['First Name','Last Name','Email','Phone Number','Address','Position','Company']
            if col in identity_col:
                print(f"{col}→ Filling with 'Unknown'")
                new_df[col] = new_df[col].fillna("Unknown")
            else:
                print(f"{col}→ Filling with mode")
                new_df[col] = new_df[col].fillna(new_df[col].mode()[0])
        

    return new_df

def handle_missing_values(df):
    new_df = df.copy()

    for col in new_df.columns:

        # skip columns without missing values
        if new_df[col].isna().sum() == 0:
            continue

        # numeric columns
        if pd.api.types.is_numeric_dtype(new_df[col]):
            skewness = new_df[col].dropna().skew()
            print(f"\nHandling numeric column: {col}")
            print(f"Skewness: {skewness:.2f}")

            if abs(skewness) < 1:
                print("Using MEAN because distribution is close to normal")
                new_df[col] = new_df[col].fillna(new_df[col].mean())
            else:
                print("Using MEDIAN because distribution is skewed")
                new_df[col] = new_df[col].fillna(new_df[col].median())

        # datetime columns
        elif pd.api.types.is_datetime64_any_dtype(new_df[col]):
            print(f"\nHandling datetime column: {col}")
            print("Using forward fill")
            new_df[col] = new_df[col].fillna(method='ffill')

        # categorical / ordinal columns
        else:
            print(f"\nHandling categorical column: {col}")
            print("Using 'Unknown'")
            new_df[col] = new_df[col].fillna("Unknown")

    return new_df
            
# ===========================================DUPLICATE CHECK=========================================================

def duplicate_check(df):
    duplicate_count = df.duplicated().sum()
    return duplicate_count

def remove_duplicates(df):
    df=df.copy()
    before_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_count = df.shape[0]
    print(f"Removed {before_count - after_count} duplicate rows.")
    return df

# ===========================================OUTLIER CHECK=========================================================
def get_iqr_bounds(df, threshold=1.5):
    bounds = {}

    for col in df.select_dtypes(include=['number']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        bounds[col] = (lower_bound, upper_bound)

    return bounds


def detect_outliers(df, threshold=1.5):
    outliers = {}
    bounds = get_iqr_bounds(df, threshold)

    for column in df.select_dtypes(include=['number']).columns:
        lower_bound, upper_bound = bounds[column]

        outliers[column] = df[
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ].index.tolist()

    return outliers


def treat_outliers(df, threshold=1.5):
    new_df = df.copy()
    bounds = get_iqr_bounds(new_df, threshold)

    for column in new_df.select_dtypes(include=['number']).columns:
        lower_bound, upper_bound = bounds[column]

        print(f"\nTreating {column}")
        print(f"Clipping values outside [{lower_bound:.2f}, {upper_bound:.2f}]")

        new_df[column] = new_df[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return new_df


# ==========================================APPLY CHANGE=========================================================
# def apply_change(df, operation_func, label):
#     choice = input(f"\nFor {label}, choose: [1] new dataframe  [2] same dataframe  [skip]: ").strip().lower()

#     if choice == "1":
#         result_df = operation_func(df)
#         print(f"Created new dataframe after {label}: shape {result_df.shape}")
#         return result_df, df

#     elif choice == "2":
#         df = operation_func(df)
#         print(f"Updated same dataframe after {label}: shape {df.shape}")
#         return df, df

#     else:
#         print(f"Skipped {label}")
#         return df, df
    
    
# ==========================================INTERACTIVE FUNCTION=========================================================

def interactive_eda_menu(df):
    print("\nHow do you want to work?")
    print("1. Modify current dataset permanently")
    print("2. Create a new cleaned dataset")

    mode = input("Enter choice (1/2): ").strip()

    if mode == "1":
        current_df = df
        print("✅ Permanent mode selected")
    else:
        current_df = df.copy()
        print("✅ New cleaned dataset mode selected")

    saved_dfs = {"original": df.copy()}
    version = 1

    print("\nPreview:")
    print(current_df.head())

    
    while True:
        print("\n" + "=" * 60)
        print("INTERACTIVE EDA MENU")
        print("1. Check missing values")
        print("2. Detect & treat outliers")
        print("3. Check duplicates")
        print("4. Show saved versions")
        print("0. Exit")

        choice = input("Enter choice: ").strip()

        # ================= MISSING VALUES =================
        if choice == "1":
            print(null_check(current_df))

            if null_check(current_df)["Null Count"].sum() > 0:
                print("Missing values detected.")
                if input("Do you want to clean missing values? (yes/no): ").strip().lower() == "yes":

                    old_df = current_df.copy()
                    current_df = smart_null_handler(current_df)

                    if not old_df.equals(current_df):
                        saved_dfs[f"version_{version}_null_cleaned"] = current_df.copy()
                        version += 1
                        print("Missing values cleaned and version saved.")
                        print(current_df.shape)
                        print(df.shape)
                else:
                    print("Skipping missing value cleanup.")

        # ================= OUTLIERS =================
        elif choice == "2":
            outliers = detect_outliers(current_df)
            
            if any(len(v)>0 for v in outliers.values()):
                print("Outliers detected.")

                if input("Do you want to treat outliers? (yes/no): ").strip().lower() == "yes":

                    old_df = current_df.copy()
                    current_df = treat_outliers(current_df)

                    if not old_df.equals(current_df):
                        saved_dfs[f"version_{version}_outlier_treated"] = current_df.copy()
                        version += 1
                        print("Outliers treated and version saved.")
            else:
                print("No outliers detected.")

        # ================= Duplicates =================
        elif choice == "3":
            dup_count = duplicate_check(current_df)
            print(f"Duplicate rows: {dup_count}")

            if dup_count > 0:
                if input("Do you want to remove duplicates? (yes/no): ").strip().lower() == "yes":

                    old_df = current_df.copy()
                    current_df = remove_duplicates(current_df)

                    if not old_df.equals(current_df):
                        saved_dfs[f"version_{version}_duplicates_removed"] = current_df.copy()
                        version += 1
                        print("Duplicates removed and version saved.")
            else:
                print("No duplicates found.")

        # ================= SAVED VERSIONS =================
        elif choice == "6":
            for name, temp_df in saved_dfs.items():
                print(f"{name}: {temp_df.shape}")

        # ================= EXIT =================
        elif choice == "0":
            print("Exiting EDA menu...")
            break

        else:
            print("Invalid choice")

    return current_df, saved_dfs


# ============================================ MAIN =========================================
if __name__ == "__main__":
    df, file_path = load_file("data/linkedin_data.csv")
    print(f"Loaded file: {file_path}")

    final_df, versions = interactive_eda_menu(df)
    print("\nFinal dataframe shape:", final_df.shape)

# # =============================================MAIN FUNCTION=========================================================
# if __name__ == "__main__":
#     df, file_path = load_file('data/bank_customers.csv')
#     print(f"Loaded file: {file_path}")
#     print("Basic Structure Check:")
#     print(basic_structure_check(df))
#     print("\nColumn Types:")
#     print(col_types(df))
#     print("\nNull Check:")
#     print(null_check(df))
#     print("\nHandling Missing Values (mean strategy):")
#     a=input("Do you want to handle missing values? (yes/no): ")
#     if a.lower() == 'yes':
#         df_cleaned = handle_missing_values(df)
#     else:
#         df_cleaned = df
#     print(df_cleaned.head())
#     print(null_check(df_cleaned))
    
  