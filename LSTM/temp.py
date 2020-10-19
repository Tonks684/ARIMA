# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.tools as tls

## LOADING file
# load the times series (2D)
Matrix = pd.read_csv("C:/Users/nt385534/Downloads/Covid-19/LSTM.csv",parse_dates =["final date"], index_col=["final date"])
dataset = np.array(Matrix);

# choose a number of time steps
n_steps_in, n_steps_out = 4, 4
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mae')
model.summary()
# fit model
model_hist = model.fit(X, y, epochs=2000, batch_size = 15)
   
def loss_mae_epoch(model_final2):
    loss_f = plt.figure()
    plt.plot(model_final2.history['loss'])
    plt.title('Loss Function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    # plot mean average error vs epoch
    mae_f = plt.figure
    plt.plot(model_final2.history['mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.show()
    
    return loss_f, mae_f

loss_mae_epoch(model_hist)

# demonstrate prediction
x_input = dataset[dataset.shape[0]-n_steps_in:,0:dataset.shape[1]];
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
