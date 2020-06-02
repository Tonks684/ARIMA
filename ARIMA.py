"""
Created Feb-19
@author: Nikos Tsiaparas

Description: This is a dashboard used for visualizing Predictions & Forecasting Diagnostics. 
The Python code for dashboard has been created by Niko and the ARIMA python model was based on external resourses

"""

###############################################################################
## INSTALLATION ##
# pip install dash
# pip install --upgrade dash dash-core-components dash-html-components dash-renderer plotly
###############################################################################

###############################################################################
## IMPORT ##
# for model
from pandas import Series
from pandas import DataFrame
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.tools as tls
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.stats import boxcox
from statsmodels.graphics.gofplots import qqplot
from math import log
from math import exp
import pandas as pd
import numpy
# for dash
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

###############################################################################
## PREPARING ARIMA## 
## A - load data & define train/test sets ##
series = Series.from_csv('sample.csv', header=0)
# split data into training and test datasets by using 70/30 ratio
split_point = int(len(series) * 0.70)
trainset, testset = series[0:split_point], series[split_point:]
trainset.to_csv('trainset.csv')
testset.to_csv('testset.csv')
## B - perform a descriptives/explanatory analysis on train data ##
# define where to run analysis (trainset or original series?)
finalseries = Series.to_frame(series)
mpl_fig1=pyplot.figure()
pyplot.subplot(211)
finalseries.hist(ax=pyplot.gca())
pyplot.subplot(212)
finalseries.plot(kind='kde', ax=pyplot.gca(),fontsize='10')
plotly_fig1 = tls.mpl_to_plotly(mpl_fig1) 
dfsummary = series.describe()
dfsummary = Series.to_frame(dfsummary)
dfsummary = dfsummary.T
dfsummary = round(dfsummary,2)
# create a fnction to evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.70)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# create a function to evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	return best_cfg, best_score

# evaluate parameters
p_values = range(0,5)
d_values = range(0, 5)
q_values = range(0, 5)
import warnings
warnings.filterwarnings("ignore")
best_cfg,best_score = evaluate_models(series.values, p_values, d_values, q_values)
(order, value) = best_cfg,best_score
value = round(value,3)
## B - evaluation on the testset by plotting residuals for best ARIMA model##
train = series.from_csv('trainset.csv', header=-1)
test = series.from_csv('testset.csv', header=-1)
train = train.values
train = train.astype('float32')
test = test.values
test = test.astype('float32')
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=order)
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
mpl_fig2=pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca(),fontsize='10')
#(the below command works with plotty: pip install plotly==2.7.0, unistall plotty version first)
plotly_fig2 = tls.mpl_to_plotly(mpl_fig2) 
# ACF and PACF plots of forecast residual errors (best model)
mpl_fig3=pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
plotly_fig3=tls.mpl_to_plotly(mpl_fig3)
mpl_fig4=pyplot.figure()
pyplot.subplot(111)
residuals.plot(ax=pyplot.gca())
plotly_fig4 = tls.mpl_to_plotly(mpl_fig4) 
## C - check transformed data ##
transformed, lam = boxcox(series)
transformed = pd.Series(transformed)
mpl_fig5=pyplot.figure()
pyplot.subplot(111)
qqplot(series, line='r', ax=pyplot.gca())
plotly_fig5=tls.mpl_to_plotly(mpl_fig5)
mpl_fig6=pyplot.figure()
pyplot.subplot(111)
qqplot(transformed, line='r', ax=pyplot.gca())
plotly_fig6=tls.mpl_to_plotly(mpl_fig6)
## D - finalize model and save to file with workaround ##
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__

# transform data if necessary
# fit model
model = ARIMA(transformed.values, order=order)
model_fit = model.fit(disp=0)
# save model
model_fit.save('model.pkl')
numpy.save('model_lambda.npy', [lam])
# load the finalized model and make a prediction
model_fit = ARIMAResults.load('model.pkl')
lam = numpy.load('model_lambda.npy')
yhat = model_fit.forecast()[0]
# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)

# use yhat = yhat if no box-cox transformation took place few steps above
yhat = boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)
## G - evaluate the finalized model on the validation dataset ##
# load and prepare datasets
series = series.values.astype('float32')
history = [x for x in series]
y = test
# load model
model_fit = ARIMAResults.load('model.pkl')
lam = numpy.load('model_lambda.npy')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
# use that if box-cox has been applied
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# transform
	transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1
	# predict
	model = ARIMA(transformed, order=order)
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	# invert transformed prediction
	yhat = boxcox_inverse(yhat, lam)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	#print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
rrmse = round(rmse,2)
#print('RMSE: %.3f' % rmse)
mpl_fig7=pyplot.figure()
pyplot.subplot(111)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
plotly_fig7=tls.mpl_to_plotly(mpl_fig7)


app = dash.Dash()   

app.layout = html.Div(children=[html.H1('Welcome to Forecasting Web Application Tool'),
                           
                                html.Div([dcc.Upload(id='upload-data',children=html.Div(['Drag and Drop or ',html.A('Select File')]),style={
                                            'width': '25%',
                                            'height': '25px',
                                            'lineHeight': '15px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        }, multiple=False),html.Div(id='output-data-upload'),]),
    
                                html.H2('Summary Descriptives'),dash_table.DataTable(id='table',columns=[{"name": i, "id": i} for i in dfsummary.columns],
                                
                                                     data=dfsummary.to_dict("rows"),style_cell={'textAlign': 'center'}),
                                 
                                html.Div([html.H2('Series Plot'),dcc.Graph(id='scatter_chart',figure={'data': [{'x' : finalseries.index, 'y' : finalseries.data, 'type': 'line'},]})]),
                                                     
                                html.Div([
                                    html.Div([
                                        html.Div([
                                            html.H2('Histogram/Density'),
                                            dcc.Graph(id='serieshist',figure = plotly_fig1)
                                        ], className="six columns"),
                                
                                        html.Div([
                                            html.H2('BoxPlot'),
                                            dcc.Graph(id='boxplot',figure=go.Figure(data=[go.Box(y=series,name = "Whiskers and Outliers",boxpoints = 'outliers')]))
                                        ], className="six columns"),
                                    ], className="row")
                                ]),
                               
                                html.Div(html.H2('Grid Search for ARIMA')),
    
                                html.Div([
                                    html.Div([
                                        html.Div([html.H5('Lag order'),
                                            dcc.RangeSlider(marks={i: 'obs {}'.format(i) for i in range(-1, 10)},min=0, max=10, value=[0, 5])
                                        ], className="four columns"),
                                
                                        html.Div([
                                            html.H5('Degree of differencing'),
                                            dcc.RangeSlider(marks={i: 'times {}'.format(i) for i in range(-1, 10)},min=0, max=10, value=[0, 5])
                                        ], className="four columns"),

                                        html.Div([
                                            html.H5('Order of moving average window'),
                                            dcc.RangeSlider(marks={i: 'size {}'.format(i) for i in range(-1, 10)},min=0, max=10, value=[0, 5])
                                        ], className="four columns"),
                                    ], className="row")
                                ]), 
    
                                html.Div(html.H3('- - - - - - - - - - - - - - - - - - ')),                            
    
                                html.Div(html.H5('ARIMA Best Model: Order = {} - RMSE = {}'.format(order,value))),
                                
                                html.Div(html.H3('- - - - - - - - - - - - - - - - - - ')),

                                html.Div(html.H2('Split Series & Run Model on Trainset')),

                                html.Div([
                                    html.Div([
                                        html.Div([html.H2('Trainset'),
                                            dcc.Graph(id='scatter_chart_train',figure={'data': [{'x' : Series.to_frame(trainset).index, 'y' : Series.to_frame(trainset).data, 'type': 'line'},]})
                                        ], className="six columns"),
                                
                                        html.Div([
                                            html.H2('Testset'),
                                            dcc.Graph(id='scatter_chart_test',figure={'data': [{'x' : Series.to_frame(testset).index, 'y' : Series.to_frame(testset).data, 'type': 'line'},]})
                                        ], className="six columns"),
                                    ], className="row")
                                ]),
    
                                html.Div(html.H2('Diagnostics on Testset Residuals for Best Model')),

                                html.Div([
                                    html.Div([
                                        html.Div([
                                                html.H2('Residuals'),dcc.Graph(id='scatter_res',figure = plotly_fig4)
                                        ], className="four columns"),
                                
                                        html.Div([
                                            html.H2('Histogram & Density'),dcc.Graph(id='scatter_res_hist',figure = plotly_fig2)
                                        ], className="four columns"),
    
                                        html.Div([
                                            html.H2('Autocorrelation'),dcc.Graph(id='scatter_res_autocorr',figure = plotly_fig3)
                                        ], className="four columns"),
                                    ], className="row")
                                ]),
    
                                html.Div(html.H2('Quality Check on QQ plots: Series vs Transformed Series')),                            
    
                                html.Div([
                                    html.Div([
                                        html.Div([
                                                html.H2('Series'),dcc.Graph(id='series',figure = plotly_fig5)
                                        ], className="six columns"),
                                
                                        html.Div([
                                            html.H2('Tranformed Series (box-cox)'),dcc.Graph(id='transformed',figure = plotly_fig6)
                                        ], className="six columns"),
                                    ], className="row")
                                ]),
                                
                                html.Div(html.H2('Final Model')),                            
    
                                html.Div([html.H5('Run on:'),dcc.RadioItems(options=[
                                        {'label': 'Series', 'value': 'Series'},
                                        {'label': 'Transformed', 'value': 'Transformed'}],value='Transformed')]),
    
                                html.Div(html.H5('ARIMA Final Model RMSE = {}'.format(rrmse))),
                                
                                html.Div([html.H2('Actual vs Predicted (red)'),
                                          dcc.Graph(id='actualvspredicted',figure = plotly_fig7)])
    
                                ]
    )
                                          
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})                                       
                                      
if __name__ == '__main__':
    app.run_server(debug=True, port=8001)