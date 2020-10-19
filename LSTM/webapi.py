# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:10:03 2020

@author: nt385534
"""


###############################################################################
## INSTALLATION ##
# pip install dash dash-renderer dash-html-coponents dash-core-components plotly
# pip install --upgrade dash dash-core-components dash-html-components dash-renderer
# pip install dash==0.38.0
# pip uninstall plotly
# pip install plotly==2.7.0
###############################################################################

###############################################################################
## IMPORT ##
import dash
import dash_core_components as dcc
import dash_html_components as html
###############################################################################


def webapi(plot1, plot2):

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
        
                                    html.Div([html.H2('Actual vs Predicted (red)'),
                                              dcc.Graph(id='actualvspredicted',figure = plot1)]),
    
                                    html.Div([html.H2('Actual vs Predicted (red)'),
                                              dcc.Graph(id='actualvspredicted',figure = plot2)])])
                                          
    if __name__ == '__main__':
        app.run_server(port=8000)
    
    return app
    