
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense


def baseline_model(numberoffeatures,timesteps,stepsout):
    
    model_baseline = Sequential()
    model_baseline.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(timesteps, numberoffeatures)))
    model_baseline.add(LSTM(100, activation='relu'))
    model_baseline.add(Dense(stepsout)) #removed initialisers
    model_baseline.compile(loss="mae", optimizer='adam', metrics= ["mse"])
    model_baseline.summary()
        
    return model_baseline



def basemodelgridsearch(numberoffeatures,stepsin,stepsout,train_x,train_y):
    
    model = KerasRegressor(build_fn=baseline_model,numberoffeatures=numberoffeatures,timesteps=stepsin,stepsout=stepsout,verbose=2)
    epoch = np.array([100,500])
    param_grid = dict(epochs=epoch)
    grid = GridSearchCV(estimator=model,param_grid=param_grid)
    grid_result=grid.fit(train_x,train_y)
    # summarise gridsearch results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params): 
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # extract best model parameters
    #epoch_num = tuple(grid_result.best_params_.values())[1]
    #optimizer =  tuple(grid_result.best_params_.values())[2]
    
    return grid_result.best_params_.values