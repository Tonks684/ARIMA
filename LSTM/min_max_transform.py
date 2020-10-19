"""
MinMax Transform ((0,1),(-1,1))
Input: Dataframe
Process: Apply MinMax for specificed feature range. Feature range being either ((0,1),(-1,1)) to columns which require transformation
Output: Normalised dataframe and scaler 
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def min_max_transform(data,feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    columns = list(data.columns)
    Output = np.array(data)
    x_size, y_size = data.shape 
    minmaxfeaturemap =[]
    Outputminmaxnormalizedbymean = np.zeros((x_size,y_size))
    for i in range(0,y_size):
    
        if y_size != 1:
            Output = pd.DataFrame(Output)
            feature_min = Output.iloc[:,i].min()
            feature_max = Output.iloc[:,i].max()
        else:
            feature_min = Output.min()
            feature_max = Output.max()

        if (feature_min >= 0 and feature_max <= 1) and y_size !=1:
            Outputminmaxnormalizedbymean = pd.DataFrame(Outputminmaxnormalizedbymean)
            Outputminmaxnormalizedbymean.iloc[:,i] = Output.iloc[:,i]
            print("Feature at Index " + str(i) + " No Transform required.")
            
        elif (feature_min < 0 or feature_max > 1) and y_size ==1:
            Outputminmaxnormalizedbymean = scaler.fit_transform(Output)
            Outputminmaxnormalizedbymean = pd.DataFrame(Outputminmaxnormalizedbymean)
            minmaxfeaturemap.append(i)
            print("Single Series Complete for Multi-step!")
            
        elif (feature_min < 0 or feature_max > 1) and y_size !=1:
            Output_df = np.array(Output.iloc[:,i])
            Output_reshape = Output_df.reshape(-1,1)
            Outputminmaxnormalizedbymean = pd.DataFrame(Outputminmaxnormalizedbymean)
            Outputminmaxnormalizedbymean_i = scaler.fit_transform(Output_reshape)
            Outputminmaxnormalizedbymean.iloc[:,i] = Outputminmaxnormalizedbymean_i
            minmaxfeaturemap.append(i)
            print("Feature at Index " + str(i) + " Transform Complete.")
    
    Outputminmaxnormalizedbymean.columns = columns
    Outputminmaxnormalizedbymean = pd.DataFrame(Outputminmaxnormalizedbymean)
   
    if y_size != 1:
        Outputminmaxnormalizedbymean_array = Outputminmaxnormalizedbymean.iloc[:,:].values
    else:
        Outputminmaxnormalizedbymean_array = Outputminmaxnormalizedbymean.iloc[:,0].values
   
    return Outputminmaxnormalizedbymean_array, Outputminmaxnormalizedbymean, scaler