"""
Sequence Creation for LSTM

"""

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
# split data to sequences - i.e. define time steps
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out,numberoffeatures,model_type):

    X, y = list(), list()
    for i in range(len(sequences)): 
        # find the end of this pattern
        end_ix = i + n_steps_in 
        ## check if we are beyond the dataset
        # gather input and output parts of the pattern  

        ## Multi-step
        if model_type == 'ms':			
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences):
                break 
            
            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
            
        ## Multi-variate
        elif model_type == 'mv':         
            out_end_ix = end_ix + n_steps_out-1
            if out_end_ix > len(sequences):
                break 
            
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1:]
            X.append(seq_x)
            y.append(seq_y)
        
        ## Multi-Parrallel
        elif model_type == 'mp':
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(sequences):
                break
		## gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]      
            X.append(seq_x)
            y.append(seq_y)
        
         
          
    X, y = np.array(X), np.array(y)
   
    if model_type != 'mp':
        X = X.reshape(X.shape[0], X.shape[1], numberoffeatures)
    
    if model_type == 'mv':
        y = y.reshape(y.shape[0], y.shape[1])

    return X,y
