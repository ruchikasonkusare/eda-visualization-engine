import pandas as pd
from itertools import combinations
from app.services.cleaning_service import col_types

# ======================INSIGHT HELPER=====================

def numeric_insight(series:pd.Series)->str:
    skew=series.dropna().skew()
    
    if skew>1:
        return "Highly right-skewed distribution"
    
    elif skew>0.5:
        return "Moderately right-skewed"
    
    elif skew<-1:
        return "Highly left-skewed distribution"
    
    elif skew<-0.5:
        return "Moderately left-skewed"
    
    else:
        return "Approximately normal distribution"
    
    
def correlation_insight(x:pd.Series,y:pd.Series)->str:
    corr=x.corr(y)
    
    if pd.isna(corr):
        return "Correlation could not be determined"
    
    if corr > 0.7:
        return f"Strong positive correlation ({corr:.2f})"
    
    elif corr>0.3:
        return f"Moderately positive correlation ({corr:.2f})"
    
    elif corr<-0.7:
        return f"Strong negative correlation ({corr:.2f})"
    
    elif corr<-0.3:
        return f"Moderately negative correlation ({corr:.2f})"
    
    else:
        return f"Weak Correlation ({corr:.2f})"


def recommendation_visulaizations(df:pd.DataFrame):
    type_info = col_types(df)
    
    inferred_map={
        item["Column"]:item["Inferrd_Type"]
        for item in type_info
    }
    print(type_info)
    
    recommendation=[]
    
    numeric_cols=[
        c for c,t in inferred_map.items()
        if t=='numeric'
    ]
    
    categorical_cols=[
        c for c,t in inferred_map.items()
        if t in ["categorical","ordinal","binary"]
    ]
    
    datetime_cols=[
        c for c,t in inferred_map.items()
        if t == 'datetime'
    ]
    
    # ====1 COLUMNS==========
    for col in df.columns:
        inferred=inferred_map[col]
        charts=[]
        
        if inferred == "identifier":
            continue
        
        if inferred == 'numeric':
            charts=[
                {
                    'chart':'histogram',
                    'reason':'Shows distribution of numeric values',
                    'insight':numeric_insight(df[col])
                },
                {
                    'chart':'boxplot',
                    'reason':'Shows spread and outliers',
                    'insight':f'Median = {df[col].median():.2f}'
                },
                {
                    'chart':'violin',
                    'reason':'Shows density and spread',
                    'insight':numeric_insight(df[col])
                },
            ]
            
        elif inferred in ["categorical","ordinal","binary"]:
            top_category=df[col].mode().iloc[0] if not df[col].mode().empty else None
            
            charts=[
                {
                    'chart':'bar',
                    'reason':'Best for frequency comparision',
                    'insight':f"MOde frequency = {top_category}"
                },
                {
                    'chart':'pie',
                    'reason':'Shows proportion of category',
                    'insight':f'{df[col].nunique()} unique categories'
                },
            ]
        
        elif inferred == 'datetime':
            charts = [
                {
                    'chart':'line',
                    'reason':'Best for time progression',
                    'insight':'Time based trend visualization'
                }
            ]
        
        recommendation.append({
            'columns':[col],
            'inferred_datatype':inferred,
            'recommended_charts':charts
        })
        
    #=============2 COLUMNS=================
    
    for col1,col2 in combinations(df.columns,2):
        type1=inferred_map[col1]
        type2=inferred_map[col2]
        charts=[]
        
        # numeric+numeric
        if type1=="numeric" and type2=="numeric":
            charts= [
                {
                    "chart":"scatter",
                    "reason":"Best for numeric relationship",
                    "insight":correlation_insight(df[col1],df[col2])
                },
                {
                    "chart":"line",
                    "reason":"Useful for ordereed proportion",
                    "insight":correlation_insight(df[col1],df[col2])
                }
            ] 
        
        # categorical+numeric
        elif (type1=="numeric" and type2 in ["categorical", "ordinal", "binary"]
              ) or (type2=="numeric" and type1 in ["categorical","ordinal","binary"]) :
            charts= [
                {
                    "chart":"boxplot",
                    "reason":"Compares numeric spread across groups",
                    "insight":"Useful for category-based spread comparision"
                },
                {
                    "chart":"bar",
                    "reason":"Shows aggregated category average",
                    "insight":"Compare group means"
                }
            ] 
            
        # datetime+numeric
        elif (type1=="numeric" and type2=="datetime"
              ) or (type2=="numeric" and type1=="datetime") :
            charts= [
                {
                    "chart":"line",
                    "reason":"Best for time trends",
                    "insight":"Trend over time detected"
                },
                {
                    "chart":"area",
                    "reason":"Good for cumulative trends",
                    "insight":"Highlight time-based growth"
                }
            ] 

        # categorical + categorical
        elif (
            type1 in ["categorical", "ordinal", "binary"]
            and type2 in ["categorical", "ordinal", "binary"]
        ):
            cross = pd.crosstab(df[col1], df[col2])
        
            strongest_pair = cross.stack().idxmax()
            strongest_count = int(cross.max().max())
        
            charts = [
                {
                    "chart": "stacked_bar",
                    "reason": "Best for comparing category composition",
                    "insight": (
                        f"Highest frequency pair = "
                        f"{strongest_pair[0]} + {strongest_pair[1]} "
                        f"({strongest_count})"
                    )
                },
                {
                    "chart": "grouped_bar",
                    "reason": "Side-by-side comparison of categories",
                    "insight": (
                        f"{col1} has {df[col1].nunique()} groups "
                        f"across {col2}"
                    )
                },
                {
                    "chart": "heatmap",
                    "reason": "Shows density of category combinations",
                    "insight": "Useful for spotting strongest category intersections"
                }
            ]
        
        if charts:
            recommendation.append({
                "columns":[col1,col2],
                "inferred_datatype":f"{type1}+{type2}",
                "recommended_charts":charts
            })
            
        
    # ==============3 COLUMN====================
            
    for cols in combinations(df.columns, 3):
        col1, col2, col3 = cols
        type1 = inferred_map[col1]
        type2 = inferred_map[col2]
        type3 = inferred_map[col3]
    
        if "identifier" in [type1, type2, type3]:
            continue
        
        charts = []
    
        types = [type1, type2, type3]
    
        # categorical + categorical + numeric
        if (
            sum(t in ["categorical", "ordinal", "binary"] for t in types) == 2
            and types.count("numeric") == 1
        ):
            charts = [
                {
                    "chart": "grouped_boxplot",
                    "reason": "Best for comparing numeric spread across 2 categories",
                    "insight": "Useful for subgroup comparison"
                },
                {
                    "chart": "grouped_bar",
                    "reason": "Compares average numeric values by category groups",
                    "insight": "Useful for hierarchical comparison"
                }
            ]
    
        # datetime + categorical + numeric
        elif (
            "datetime" in types
            and "numeric" in types
            and any(t in ["categorical", "ordinal", "binary"] for t in types)
        ):
            charts = [
                {
                    "chart": "multi_line",
                    "reason": "Shows trends over time across groups",
                    "insight": "Useful for segmented time-series analysis"
                },
                {
                    "chart": "stacked_area",
                    "reason": "Shows composition change over time",
                    "insight": "Useful for contribution trends"
                }
            ]
    
        # numeric + numeric + categorical
        elif (
            types.count("numeric") == 2
            and any(t in ["categorical", "ordinal", "binary"] for t in types)
        ):
            charts = [
                {
                    "chart": "colored_scatter",
                    "reason": "Shows numeric relationship separated by category",
                    "insight": "Useful for grouped correlation analysis"
                }
            ]
    
        if charts:
            recommendation.append({
                "columns": list(cols),
                "inferred_datatype": " + ".join(types),
                "recommended_charts": charts
            })
            
            
    # =============3+ CoLUMNS=========
        
    if len(numeric_cols)>3:
        recommendation.append({
            "columns":numeric_cols[:5],
            "inferred_datatype":"multi-numeric",
            "recommended_charts":[
                {
                    "chart":"pairplot",
                    "reason":"Best for multiple numeric relationship",
                    "insight":"Useful for multi-feature comparision"
                },
                {
                    "chart":"heatmap",
                    "reason":"Shows correlation matrix",
                    "insight":"Find strongest correlated features"
                }
            ]
        })
    
    return recommendation