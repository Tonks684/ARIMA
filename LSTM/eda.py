# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:16:33 2020

@author: smt29021
"""

###############################################################################
## INSTALLATION ##
# ++ all libraries needed 
###############################################################################


###############################################################################
## IMPORT PACKAGES ##
# fundamental packages
import numpy as np
from numpy import array
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.special import softmax
#from sklearn.model_selection import GridSearchCV
import pandas as pd
# deep learning packages
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
#from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model

##################### Exploratory Data Analysis Class #########################

from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import plotly.tools as tls

class EDA():
    
    def __init__(self,data):
        self.data = data
        self.num_cols = len(data.columns)
        self.data_to_search = data.columns
        self.num_vars = [var for var in self.data_to_search \
                         if self.data[var].dtypes != 'O']
        self.var = [var for var in self.data_to_search]
    
   # def na_check(self):
   #     for ind, v in enumerate(self.var):
   #         arr = np.array(self.data)
   #         if np.isnan(arr[:,ind]).any() == True:
    #            return (v + " Contains Null Value(s). Please reformat!")
     #       else:
      #          return "No NAs found for all Feature Columns!"
        
    def descriptives(self):
        print('Number of numerical variables: ', len(self.num_vars))
        data4Descriptives = self.data[self.num_vars]
        kurt = kurtosis(data4Descriptives)
        skewn = skew(data4Descriptives)
        Descriptives = data4Descriptives.describe()
        Descriptives = pd.DataFrame(Descriptives)
        Descriptives.loc['kurt'] = kurt
        Descriptives.loc['skew'] = skewn
        return Descriptives


#What is the purpose of this function?
        
    def analyse_rare_labels(self):
        dict_labels = {}        
        for ind, v in enumerate(self.var):
            tmp = 100*self.data.groupby(v)[self.data_to_search.values[ind]].count()\
                / len(self.data) 
            total_tmp = tmp.sum()
            if (total_tmp <= 99.99999999 or total_tmp >= 100.0000000001):
               # sum_temp = str(v)+ ' Feature: Sum is not equal to 100% - there are missing values'
                dict_labels["missing_labels"] =  v
                return dict_labels
            elif:
                sum_temp = str(v)+ ' Sum is equal to 100% - there are no missing values'
                return sum_temp
            else:
                
 
 #Need to add into analyse_rare_labels           
 # make a list of the variables that contain missing values
     #    varis_with_na = [varis for varis in self.data_to_search\
      #                          if self.data[var].isnull().sum()>0]
             
        # print the variable name and the percentage of missing values
       # for var in vars_with_na:
        #    print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')
        



    def analyse_discrete(self):
        dictionary_line_figures={}
        for ind, v in enumerate(self.var):
            line_plot_figure = plt.figure()
            self.data[self.data_to_search.values[0]].plot(label=\
                                self.data_to_search.values[0], legend=True)
            self.data[v].plot(secondary_y=True, label=v, legend=True)
            plt.show()
            line_plot = tls.mpl_to_plotly(line_plot_figure)
    #Dictionary of figures: Need to isolate what I need.
            dictionary_line_figures[v] = line_plot
            return dictionary_line_figures

    def analyse_continous(self):
        for ind, v in enumerate(self.var):
            hist_plot_figure = plt.figure()
            self.data[v].hist(bins=20)
            plt.ylabel(v)
            plt.xlabel('Bin range')
            plt.title(v)
            plt.show()
            histogramplot = tls.mpl_to_plotly(hist_plot_figure)
            return histogramplot
    
        # dictionary of figures
   # dictionary_histogram_figures = {}    
    #for var in num_vars:
     #   histogramplot = analyse_continous(data, var)
      #  dictionary_histogram_figures[var] = histogramplot
        
    
    # scattergrams
    def transform_analyse_continous(self):
        for ind, v in enumerate(self.var):
            scatter_plot_figure = plt.figure()
            plt.scatter(self.data[v], self.data[self.data_to_search.values[0]])
            plt.ylabel(self.data_to_search.values[0])
            plt.xlabel(v)
            plt.show()
            scatterplot = tls.mpl_to_plotly(scatter_plot_figure)
            return scatterplot
    
    # dictionary of figures
   # dictionary_scatter_figures = {}        
   # for var in num_vars:
   #     if var !=data.columns.values[0]:
   #         scatterplot = transform_analyse_continous(data, var)
   #         dictionary_scatter_figures[var] = scatterplot
          
     # boxplots outliers in the continuous variables 
    def find_outliers(self):
        for ind, v in enumerate(self.var):
            box_plot_figure = plt.figure()
            self.data.boxplot(column=v)
            plt.show()
            box_plot = tls.mpl_to_plotly(box_plot_figure)
            return box_plot
    # dictionary of figures
    # dictionary_boxplot_figures = {}            
    # for var in num_vars:
    #    box_plot = find_outliers(data, var)
    #    dictionary_boxplot_figures[var] = box_plot


################# Normalise & Softmax Scaling #################################
    
from sklearn.base import TransformerMixin, BaseEstimator

class SoftScaler(TransformerMixin,BaseEstimator):
    
    def __init__(self,data):
        self.data = data
        self.data_to_search = data.columns
        self.var = [var for var in self.data_to_search]
        self.is_fitted = False

    def fit(self,X,y=None):
        self.mean = np.mean(X,axis=0)
        self.is_fitted = True
        return self, "Fit Complete!"
        
    def transform(self,X,y=None):
        
        if not self.is_fitted:
            raise ValueError(
                "The data needs to be fitted before it is transformed")
            
        for v in self.var:
            Xt = self.data[v]/self.data[v].mean()
            s_Xt = softmax(Xt)
        return s_Xt

################# Apply Transform to each Col #################################
            
from sklearn.compose import ColumnTransformer
def column_trans(matrix):
    all_cols = matrix.columns.values
    soft_cols = []
    non_soft_cols = []
    y_size = matrix.shape[1]
    for i in range(y_size):
        if(matrix.iloc[:,i].min() >= 0 and matrix.iloc[:,i].max() <=1):
            non_soft_cols.append(all_cols[i])
        elif(matrix.iloc[:,i].min() < 0 or matrix.iloc[:,i].max() > 1):
            soft_cols.append(all_cols[i])
    return soft_cols, non_soft_cols

    
 ################# Train & Test Split Define Windows ##########################
   
def train_test_split(data,length_test,x_size,y_size, timesteps):
    length_input = x_size - 2*length_test
    stepsout = length_test
    
    if y_size == 1:
        train_set = pd.DataFrame(data.iloc[:length_input,0])
        test_set = pd.DataFrame(data.iloc[(length_input+length_test):,0])
        input_for_prediction = pd.DataFrame(data.iloc[length_input:length_input+length_test,0])
        x_input_train = train_set.iloc[:, 0].values.tolist()
        x_input = input_for_prediction.iloc[:, 0].values.tolist()
    else:
        train_set = pd.DataFrame(data.iloc[:length_input,:])
        test_set = pd.DataFrame(data.iloc[(length_input+length_test):,:])
        input_for_prediction = pd.DataFrame(data.iloc[length_input:length_input+length_test,:-1])
        x_input_train = np.array(train_set)
        x_input = np.array(input_for_prediction)
        
    return x_input_train, stepsout

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		# check if we are beyond the dataset
		# gather input and output parts of the pattern        
		if sequences.shape[0] == 1:			
			out_end_ix = end_ix + n_steps_out
			if out_end_ix > len(sequences):
				break 
            
			seq_x, seq_y = sequences[i:end_ix], sequences[end_ix:out_end_ix]
		else:         
			out_end_ix = end_ix + n_steps_out-1
			if out_end_ix > len(sequences):
				break 
            
			seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

