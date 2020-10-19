# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
def create_traintest_input(train_rescaled_array,test_rescaled_array, stepsin,stepsout,model_type):
    if model_type == 'mv':
        last_12_train_data_mv = pd.DataFrame(train_rescaled_array)
        last_12_train_data_mv = last_12_train_data_mv.iloc[-stepsin:, :-1]
        test_data = pd.DataFrame(test_rescaled_array)
        test_data = test_data.iloc[:, :-1]
        data_mv = last_12_train_data_mv.append(test_data)
        return data_mv
    elif model_type == 'mp':
        last_12_train_data_mv = pd.DataFrame(train_rescaled_array)
        last_12_train_data_mv = last_12_train_data_mv.iloc[-stepsin:,:]
        test_data = pd.DataFrame(test_rescaled_array)
        test_data = test_data.iloc[:, :]
        data_mp = last_12_train_data_mv.append(test_data)
        #data_mv = last_12_train_data_mv   
        return data_mp
    elif model_type == 'ms':
        data_ms = train_rescaled_array[0:stepsin]
        data_ms = data_ms.reshape(-1,1)
        return data_ms