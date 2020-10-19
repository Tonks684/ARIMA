# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:03:20 2020

@author: smt29021
"""
import pandas as pd
def data_analysis(data):
    
    data = pd.DataFrame(data)
    Num_cols = len(data.columns)
    if Num_cols == 1:
        datatosearch = data
    else:
        datatosearch = data.columns
        
    import numpy as np
    from scipy.stats import kurtosis
    from scipy.stats import skew
    # check for Numerical & Categorical variables
    # numerical variables
    num_vars = [var for var in datatosearch if data[var].dtypes != 'O']
    print('Number of numerical variables: ', len(num_vars))
    # get descriptives for numerical values
    data4Descriptives = data[num_vars]
    kurt = kurtosis(data4Descriptives)
    skewn = skew(data4Descriptives)
    Descriptives = data4Descriptives.describe()
    Descriptives = pd.DataFrame(Descriptives)
    Descriptives.loc['kurt'] = kurt
    Descriptives.loc['skew'] = skewn
    Descriptives = Descriptives.round(2)
    print(Descriptives)
    ############################################
    if Num_cols != 1:
        # categorial variables
        cat_vars = [var for var in data.columns if data[var].dtypes=='O']
        print('Number of categorical variables: ', len(cat_vars))
        # visualise the values of the categorical variables
        data[cat_vars].head()
        # check number of categories for each categorial variable
        for var in cat_vars:
            print(var, len(data[var].unique()), ' categories')
        
        # check if subcategories sum up to 100%    
        def analyse_rare_labels(df, var):
            df = df.copy()
            tmp = 100*df.groupby(var)[df.columns.values[0]].count() / len(df) 
            total_tmp = tmp.sum()
            if total_tmp != 1:
                sum_temp = 'Sum is not equal to 100% - there are missing values'
            return tmp,sum_temp
        
        for var in cat_vars:
            print(analyse_rare_labels(data, var))
            print()
        
        # make a list of the variables that contain missing values
        vars_with_na = [var for var in data.columns if data[var].isnull().sum()>0]
        # print the variable name and the percentage of missing values
        for var in vars_with_na:
            print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')
        
    # line plots
    import matplotlib.pyplot as plt
    import plotly.tools as tls
    def analyse_discrete(df, var):
        line_plot_figure = plt.figure(figsize=(20,5))
        df[df.columns.values[-1]].plot(label=df.columns.values[-1], legend=False)
        df[var].plot(secondary_y=True, label=var, legend=True)
        plt.xticks(rotation = 90)
        plt.show()
        line_plot = tls.mpl_to_plotly(line_plot_figure)
        return line_plot

    # dictionary of figures    
    dictionary_line_figures = {}
    for var in num_vars:
        lineplot = analyse_discrete(data, var)
        dictionary_line_figures[var] = lineplot
        
    # histogramms
    def analyse_continous(df, var):
        hist_plot_figure = plt.figure(figsize=(20,5))
        df = df.copy()
        df[var].hist(ax=plt.gca())
        plt.ylabel(var)
        plt.xlabel('Bin range')
        #plt.title(var)
        plt.show()
        histogramplot = tls.mpl_to_plotly(hist_plot_figure)
        return histogramplot
    
    # dictionary of figures
    dictionary_histogram_figures = {}    
    for var in num_vars:
        histogramplot = analyse_continous(data, var)
        dictionary_histogram_figures[var] = histogramplot
              
    # scattergrams
    def transform_analyse_continous(df, var):
        scatter_plot_figure = plt.figure(figsize=(20,5))
        df = df.copy()   
        plt.scatter(df[var], df[df.columns.values[-1]])
        plt.ylabel(df.columns.values[-1])
        plt.xlabel(var)
        plt.show()
        scatterplot = tls.mpl_to_plotly(scatter_plot_figure)
        return scatterplot
    
    # dictionary of figures
    dictionary_scatter_figures = {}        
    for var in num_vars:
        if var !=data.columns.values[-1]:
            scatterplot = transform_analyse_continous(data, var)
            dictionary_scatter_figures[var] = scatterplot
                      
    # boxplots outliers in the continuous variables 
    def find_outliers(df, var):
        box_plot_figure = plt.figure(figsize=(20,5))
        df = df.copy()
        df.boxplot(column=var)
        plt.show()
        box_plot = tls.mpl_to_plotly(box_plot_figure)
        return box_plot
    
    # dictionary of figures
    dictionary_boxplot_figures = {}            
    for var in num_vars:
        box_plot = find_outliers(data, var)
        dictionary_boxplot_figures[var] = box_plot

    return Descriptives, dictionary_line_figures, dictionary_histogram_figures, dictionary_scatter_figures, dictionary_boxplot_figures

