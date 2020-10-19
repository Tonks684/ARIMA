# -*- coding: utf-8 -*-
"""
Split Dataset
- Input: dataframe and split of full dataset
- Output: train and test sets

"""
import pandas as pd
def train_test_split(data,length_test): 
    x_size, y_size = data.shape
    length_input = x_size - length_test
    if y_size == 1:
        x_input_train = pd.DataFrame(data.iloc[:length_input,0])
        x_input_test = pd.DataFrame(data.iloc[length_input:,0])
    else:
        x_input_train = pd.DataFrame(data.iloc[:length_input,:])
        x_input_test = pd.DataFrame(data.iloc[length_input:,:])
    
    print('Trainset shape: ', x_input_train.shape)
    print('Testset shape: ', x_input_test.shape)
    return x_input_train, x_input_test
