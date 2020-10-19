# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:36:26 2020

@author: smt29021
"""

## IMPORT PACKAGES ##
# fundamental packages
import os
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import array
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
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
from numpy import hstack
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

