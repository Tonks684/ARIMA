# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:21:28 2020

@author: smt29021

Aim: 
    - Import all individual actual numbers from 2016:Current Date
    - Import predictions based on specific forecast_horizon
    - Create YOY Figures and Comparisons

Variables:

#datafiles = list of file names
#Start Date = start date of actual series
#Weekly - Binary variable for weekly vs monthly
#categories = category names as list in same order as datafiles


Data:

Assumed to be 2017 - 2019 
"""

import numpy as np
import pandas as pd
import datetime




def final_table(datafortemplate, forecasthorizon,start_date,Subcategory, Monthly):
    
    figures = np.zeros((datafortemplate.shape[0],1))
    
    matrix = datafortemplate
    figures = matrix.iloc[:,0]
    figures = pd.DataFrame(figures)
    #figures = figures.astype(int)
    if Monthly == True:
        figures["date"]= pd.date_range(start=start_date, periods=figures.shape[0], freq='M')
        figures["date"] = pd.to_datetime(figures["date"])
    else:
        figures["date"]= pd.date_range(start=start_date, periods=figures.shape[0], freq='W')
        figures["date"] = pd.to_datetime(figures["date"])
    cols = figures.columns.tolist()
    cols = cols[-1:] + cols [:-1]
    figures = figures[cols]
    #Groupby Year for each series
    for i in range((figures.shape[1])-1):
        if i == 0:
            totals = figures.groupby(figures['date'].dt.year)[i].agg(['sum'])
            totals = totals.transpose()
            
        else:
            totals_i = figures.groupby(figures['date'].dt.year)[i].agg(['sum'])
            totals_i = totals_i.transpose()
            totals = totals.append(totals_i)
            
    #Create YOY % Columns 
    #Update row values to be Category names
    totals = totals.reset_index()
    totals['Category'] = Subcategory
    cols = totals.columns
    totals["2018 MAT YOY %"] =(totals.iloc[:,3]/totals.iloc[:,2] -1)*100
    totals["2019 MAT YOY %"] =(totals.iloc[:,4]/totals.iloc[:,3] -1)*100
    totals = totals.rename(columns={2016: "MAT FY 2016", 2017: "MAT FY 2017", 2018: "MAT FY 2018", 2019: 'MAT FY 2019', 2020: "MAT 2020 FC", 2021: "MAT 2021 FC"})
    cols = totals.columns
    total_cols = cols[[6,1,2,3,7,4,8,5]]
    totals = totals[total_cols]
    totals = totals.round(1) 
    return totals

def final_figures(datafiles, forecasthorizon,actual_length,start_date,categories,Monthly):
    
    figures = np.zeros((actual_length,len(datafiles)))
    
    #Combine all files into single file
    for index, file in enumerate(datafiles):
        matrix = pd.read_csv(file)
        figures[:,index] = matrix.iloc[:,0]
    figures = pd.DataFrame(figures)
    if Monthly == True:
        figures["date"]= pd.date_range(start=start_date, periods=figures.shape[0], freq='M')
        figures["date"] = pd.to_datetime(figures["date"])
    else:
        figures["date"]= pd.date_range(start=start_date, periods=figures.shape[0], freq='W')
        figures["date"] = pd.to_datetime(figures["date"])
    cols = figures.columns.tolist()
    cols = cols[-1:] + cols [:-1]
    figures = figures[cols]
    print(figures.shape)   
    #Groupby Year for each series
    for i in range((figures.shape[1])-1):
        if i == 0:
            totals = figures.groupby(figures['date'].dt.year)[i].agg(['sum'])
            totals = totals.transpose()
            #print(totals.shape)
            
        else:
            totals_i = figures.groupby(figures['date'].dt.year)[i].agg(['sum'])
            totals_i = totals_i.transpose()
            totals = totals.append(totals_i)
            #print(totals.shape)
    
    #Create YOY % Columns 
    #Update row values to be Category names
    totals = totals.reset_index()
    print(totals.columns)
    #categories = {0:"test",1:"test1",2:"test2",3:"test3",4:"test4", 5:"test5"}
    totals['Category'] = categories.values()
    totals["2018 MAT YOY %"] =(totals.iloc[:,3]/totals.iloc[:,2] -1)*100
    totals["2019 MAT YOY %"] =(totals.iloc[:,4]/totals.iloc[:,3] -1)*100
    #May need to add above for 2020
    #cols = list(totals.columns.values)
    #print(cols)
    #print(cols)
    cols = ["Category", 2016,2017,2018, "2018 MAT YOY %", 2019, "2019 MAT YOY %"]
    #cols = cols[5]+ cols[1] + cols[2] +cols[3] + cols[6] + cols[4]+ cols[7]
    totals = totals[cols]
    return totals

    
    #passed_weeks_latest_year = totals_count.iloc[-1].values
    #totals ["2020 MAT FC"]= totals["2020"] + data_test_predictions.iloc[past_weeks_latest_year: forecasthorizon,:]
