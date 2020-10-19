# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:35:16 2020

@author: smt29021
"""
import pandas as pd
# deep learning packages
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


# MODEL3: FINAL: REGULARIZATION (L2) + DROPOUT
def final_model(timesteps,numberoffeatures,stepsout,epoch_num,train_x,train_y,dp_value,batch_size,model_type):
    if model_type == 'ms':
        model_final = Sequential()
        model_final.add(LSTM(10, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
        model_final.add(Dropout(dp_value))
        model_final.add(LSTM(10, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
        model_final.add(Dropout(dp_value))
        model_final.add(LSTM(10, activation = "relu",kernel_regularizer=regularizers.l2(0.001)))
        model_final.add(Dropout(dp_value))
        model_final.add(Dense(stepsout)) #removed initialisers
        model_final.compile(loss="mae", optimizer='adam', metrics= ["mae"])
        model_final.summary()
        model_final2 = model_final.fit(train_x,train_y, epochs=epoch_num, batch_size=batch_size)
       
        return model_final, model_final2
 
    elif model_type == 'mv':
        model_final = Sequential()
        model_final.add(LSTM(10, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
        model_final.add(Dropout(dp_value))
        model_final.add(LSTM(10, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
        model_final.add(Dropout(dp_value))
        model_final.add(LSTM(10, activation = "relu",kernel_regularizer=regularizers.l2(0.001)))
        model_final.add(Dropout(dp_value))
        model_final.add(Dense(stepsout)) #removed initialisers
        model_final.compile(loss="mae", optimizer='adam', metrics= ["mae","mape"])
        model_final.summary()
        model_final2 = model_final.fit(train_x,train_y, epochs=epoch_num, batch_size=batch_size)
        
        return model_final, model_final2
    
    elif model_type == 'mp':
    
        model = Sequential()
        model.add(LSTM(10, activation='relu', input_shape=(timesteps, numberoffeatures)))
        model.add(RepeatVector(stepsout))
        model.add(Dropout(dp_value))
        model.add(LSTM(10, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(numberoffeatures)))
        model.add(Dropout(dp_value))
        model.compile(optimizer='adam', loss='mae', metrics= ["mae"])
        model.summary()
        model_final = model.fit(train_x,train_y, epochs=epoch_num, batch_size=batch_size)
    
        return model, model_final