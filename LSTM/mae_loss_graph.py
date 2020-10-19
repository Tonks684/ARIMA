# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:31:58 2020

@author: smt29021
"""
import matplotlib.pyplot as plt
import plotly.tools as tls
    
def metrics_epoch(model_final2):
    loss_f = plt.figure()
    plt.plot(model_final2.history['loss'])
    plt.title('Loss Function')
    plt.ylabel('LOSS')
    plt.xlabel('EPOCH')
    plt.show()
    
    # plot mean average error vs epoch
    mae_f = plt.figure
    plt.plot(model_final2.history['mae'])
    plt.title('Mean Average Error')
    plt.ylabel('MAE')
    plt.xlabel('EPOCH')
    plt.show()

    mape_f = plt.figure
    plt.plot(model_final2.history['mape'])
    plt.title('Mean Absolute Percentage Error')
    plt.ylabel('MAPE')
    plt.xlabel('EPOCH')
    plt.show()
    
    #return loss_f, mae_f, mape_f
    