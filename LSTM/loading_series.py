# -*- coding: utf-8 -*-
"""
Loading Series

Inputs
    
    - Filename: CSV file defined in Notebook
        1. Multi-step file is assumed to contain only one subcategory and date column. Can deal with multiple series
        2. Multi-variate file is assumed to contain all features with final column as dependent variable
        3. Multi-parallel file is assumed to contain all individual series as columns with with column as total 
        
    - Subcategory: Defined in Notebook  
        1. Specific subcategory for multi-step and multi-variate models eg. 'OTC Toothpaste'
        2. For Multi-paralell: SubCategory not required
        

Output
    
    - Dataframe and the name of the final column

"""
import pandas as pd

def load_series(Filename, Subcategory,model_type):
    OriginalDataSeries = pd.read_csv(Filename,index_col ='date', parse_dates=True)
    
    if model_type == 'ms':
       OriginalDataSeries = pd.DataFrame(OriginalDataSeries[Subcategory])
       
    #get data frame's dimensions and the name of the column used for analysis
    x_size,y_size = OriginalDataSeries.shape
    columnsNamesArray = OriginalDataSeries.columns.values
    listOfColumnNames = list(columnsNamesArray)
    NameofTimeSeriesAnalysed = listOfColumnNames[-1]
    #print(OriginalDataSeries.dtypes)
    return OriginalDataSeries,NameofTimeSeriesAnalysed
