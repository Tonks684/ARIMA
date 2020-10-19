# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:13:37 2020

@author: smt29021
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers.core import Lambda
#from keras import backend as K
#from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor

def grid_search(train_sequences_stepsin,train_sequences_stepsout,numberoffeatures,stepsin,stepsout):
    
    #MODEL1 = BASEMODEL: OPTIMIZE PARAMETERS
    def base_model(numberoffeatures,stepsin,stepsout):
    
        model = Sequential()
        model.add(LSTM(100, activation = "relu", return_sequences=True, input_shape=(stepsin, numberoffeatures)))   
        model.add(LSTM(100, activation = "relu", return_sequences=True, input_shape=(stepsin, numberoffeatures)))   
        model.add(LSTM(100, activation = "relu", return_sequences=False, input_shape=(stepsin, numberoffeatures)))   
        model.add(Dense(stepsout)) #removed initialisers
        model.compile(loss="mae", optimizer='adam', metrics= ["mae"])
        model.summary()
    
        return model
    
# define width and depth of Network as well as epoch, batch size and number of output observations
    model = KerasRegressor(build_fn=base_model,numberoffeatures=numberoffeatures,stepsin=stepsin,stepsout=stepsout,verbose=2)    
    epoch = np.array([300,500,700,1000,1500])
    batch_size = np.array([1,2,3,4,6])
    param_grid = dict(epochs=epoch,batches = batch_size)
    grid = GridSearchCV(estimator=model,param_grid=param_grid)
    grid_result = grid.fit(train_sequences_stepsin,train_sequences_stepsout)
#summarise gridsearch results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params): 
        print("%f (%f) with: %r" % (mean, stdev, param))

# extract best model parameters
    epoch_num_bs = tuple(grid_result.best_params_.values())[0]
    return epoch_num_bs
