import os
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import re

        
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CYBORG])
server = app.server
port = int(os.environ.get("PORT", 5000))

app.layout = html.Div(
                  [
                    html.H2(
                             'DoE Matrix Analyzer',
                             style = {
                                       'textAlign' : 'center'
                                     }
                           ),
                    html.Br(),
                    dbc.Row(
                        [
                          dbc.Col(
                              [
                                 html.Ul(
                                      [
                                        html.Li('Two-level full factorial DoE design matrix - X'),
                                        html.Li('Size of matrix - 2^n'),
                                        html.Li('Effect of cross-interactions of variables evaluated'),
                                        html.Li('Response - Y, obtained from actual experiments')
                                      ]
                                        ),
                                 html.Br(),
                                 dbc.Label('Select number of variables to generate design matrix'),
                                 dcc.Dropdown(
                                        id = 'dd_num',
                                        options = [
                                                    {'label' : '%s' %num, 'value' : num} for num in [3, 4, 5]
                                                  ],
                                        placeholder = 'select n for matrix 2^n',
                                        value = None,
                                        style = {
                                                  'width' : '60%',
                                                  'height' : 16,
                                                  'margin-left' : '5%',
                                                  'background-color' : 'black',
                                                  'color' : 'black'
                                                }
                                             ),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Label('Display design matrix - experiment plan'),
                                 html.Br(),
                                 html.Div(
                                           id = 'matrix',
                                           style = {
                                                     'width' : '80%',
                                                     'height' : 80,
                                                     'color' : 'white',
                                                     'fontSize' : 13,
                                                     'margin-left' : '5%'
                                                   }
                                         ),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Label('Clear selection and design matrix'),
                                 html.Br(),
                                 html.Button(
                                              'Reset',
                                              id = 'bt_reset',
                                              style = {
                                                        'width' : '16%',
                                                        'height' : 28,
                                                        'margin-left' : '7.5%',
                                                        'background-color' : 'black',
                                                        'color' : 'white'
                                                      }                                              
                                            ),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Label('Input experiment result - Y'),
                                 html.Br(),
                                 dcc.Textarea(
                                               placeholder = 'Y = 2^n',
                                               id = 'Y',
                                               style = {
                                                         'width' : '50%',
                                                         'height' : 40,
                                                         'background-color' : 'black',
                                                         'margin-left' : '7.5%',
                                                         'color' : 'white',
                                                         'fontSize' : 13
                                                       }
                                             ),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Label('Prediction with slider bars'),
                                 html.Br(),
                                 html.Div(
                                           [
                                             dbc.Label('Slider for Factor A'),
                                             html.Br(),
                                             dcc.Slider(
                                                         id = 'slider_A',
                                                         min = -1,
                                                         max = 1,
                                                         step = 0.05,
                                                         tooltip = {'always_visible' : True}                                                                     
                                                       ),
                                             dbc.Label('Slider for Factor B'),
                                             html.Br(),
                                             dcc.Slider(
                                                         id = 'slider_B',
                                                         min = -1,
                                                         max = 1,
                                                         step = 0.05,
                                                         tooltip = {'always_visible' : True}                                                                     
                                                       ),
                                             dbc.Label('Slider for Factor C'),
                                             html.Br(),
                                             dcc.Slider(
                                                         id = 'slider_C',
                                                         min = -1,
                                                         max = 1,
                                                         step = 0.05,
                                                         tooltip = {'always_visible' : True}                                                                     
                                                       ),
                                             dbc.Label('Slider for Factor D'),
                                             html.Br(),
                                             dcc.Slider(
                                                         id = 'slider_D',
                                                         min = -1,
                                                         max = 1,
                                                         step = 0.05,
                                                         tooltip = {'always_visible' : True}                                                                     
                                                       ),
                                             dbc.Label('Slider for Factor E'),
                                             html.Br(),
                                             dcc.Slider(
                                                         id = 'slider_E',
                                                         min = -1,
                                                         max = 1,
                                                         step = 0.05,
                                                         tooltip = {'always_visible' : True}                                                                     
                                                       )                                     
                                           ],
                                             style = {
                                                       'width' : '50%',
                                                       'height' : 50,
                                                       'margin-left' : '5%',
                                                       'background-color' : 'black'
                                                     }
                                         ),                                
                              ],
                              lg = 6, 
                              md = 3
                                 ),
                        
                          dbc.Col(
                              [
                                 html.Ul(
                                      [
                                        html.Li('Matrix calculation to yield matrix B'),
                                        html.Li('Regression equation displayed'),
                                        html.Li('Goodness of fit, goodness of prediction displayed'),
                                        html.Li(
                                             [
                                               'Homepage -',
                                               html.A('https://linkedin.com/in/waynegu', 
                                                       href = 'https://linkedin.com/in/waynegu')
                                             ]
                                               )
                                      ]
                                        ),
                                 html.Br(),
                                 dbc.Label('Select variables for assessment'),
                                 html.Br(),
                                 html.Br(),
                                 dcc.Dropdown(
                                        id = 'dd_fcts',
                                        value = None,
                                        placeholder = 'select variables for regression',
                                        multi = True,
                                        style = {
                                                  'width' : '75%',
                                                  'height' : 20,
                                                  'margin-left' : '3%',
                                                  'background-color' : 'black',
                                                  'color' : 'black'
                                                }                                        
                                             ),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 html.Div(
                                           id = 'equation',
                                           style = {
                                                     'width' : '80%',
                                                     'height' : 50,
                                                     'color' : 'white',
                                                     'fontSize' : 13,
                                                     'margin-left' : '5%'
                                                   }                                           
                                         ),
                                 html.Div(
                                           id = 'R2',
                                           style = {
                                                     'width' : '80%',
                                                     'height' : 50,
                                                     'color' : 'white',
                                                     'fontSize' : 13,
                                                     'margin-left' : '5%'
                                                   }                                           
                                         ),
                                 html.Div(
                                           id = 'Q2',
                                           style = {
                                                     'width' : '80%',
                                                     'height' : 50,
                                                     'color' : 'white',
                                                     'fontSize' : 13,
                                                     'margin-left' : '5%'
                                                   }                                           
                                         ),
                                 html.Button(
                                              'Clear',
                                              id = 'bt_clr',
                                              style = {
                                                        'width' : '13%',
                                                        'height' : 25,
                                                        'background-color' : 'black',
                                                        'margin-left' : '5%',
                                                        'color' : 'white'
                                                      }     
                                                      
                                            ),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Label('Show predicted response below'),
                                 html.Br(),
                                 html.Br(),
                                 html.Div(
                                           id = 'response',
                                           style = {
                                                     'width' : '80%',
                                                     'height' : 50,
                                                     'color' : 'white',
                                                     'fontSize' : 20,
                                                     'margin-left' : '5%'                                               
                                                   }
                                         )
                              ],
                              lg = 6, 
                              md = 3
                                 )
                        ]  
                           )
                  ]
                     )

#  purpose of the callback below is to define and pass 'options' component to Dropdown - dd_fcts 
#  and to create the design matrix - experiment plan
#  bt_reset is used to reset the design matrix, but does not stop the information flow from dd_num to dd_fcts 
@app.callback(
              Output('dd_fcts', 'options'), 
              Output('matrix', 'children'),
              Output('bt_reset', 'n_clicks'),
              Input('dd_num', 'value'),
              Input('bt_reset', 'n_clicks')
             )
def update_option(input_num, n_clicks):
    if not input_num:
        raise PreventUpdate
        
    if input_num == 3:
        options = [{'label' : '%s' %fct, 'value' : fct} for fct in ['A', 'B', 'C', 'AB', 'AC', 'BC']]
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1]
                           }
                         )
        str_matrix = 'A B C : ' + str(np.array(df))

    elif input_num == 4:
        options = [{'label' : '%s' %fct, 'value' : fct} for fct in ['A', 'B', 'C', 'D', 'AB', 'AC', 'AD',\
                                                                    'BC', 'BD', 'CD']]
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         )
        str_matrix = 'A B C D : ' + str(np.array(df))  

    elif input_num == 5:
        options = [{'label' : '%s' %fct, 'value' : fct} for fct in ['A', 'B', 'C', 'D', 'E', 'AB', 'AC', 'AD',\
                                                                    'AE','BC', 'BD', 'BE', 'CD', 'CE', 'DE']]
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, \
                                    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, \
                                    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
                             'E' : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         ) 
        str_matrix = 'A B C D E : ' + str(np.array(df))  

    if not n_clicks:
        return options, str_matrix, None
    
    elif n_clicks:
        return options, None, None


@app.callback(
               Output('equation', 'children'),
               Output('R2', 'children'),
               Output('Q2', 'children'),
               Output('bt_clr', 'n_clicks'),
               Input('dd_num', 'value'),
               Input('dd_fcts', 'value'),
               Input('Y', 'value'),
               Input('bt_clr', 'n_clicks')
             )
def update_results(input_num, input_fcts, input_Y, n_clicks):  
    if not input_num or not input_fcts or not input_Y:
        raise PreventUpdate
    
    mo = re.compile(r'\d*\.?\d+')
    Y = np.array(mo.findall(input_Y)).astype(float).T    # Get Y for matrix calculation
    
    if 2**input_num != len(Y):
        raise PreventUpdate
        
    if input_num == 3:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1]
                           }
                         )
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['BC'] = df.B * df.C
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])

    elif input_num == 4:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         )
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['AD'] = df.A * df.D
        df['BC'] = df.B * df.C
        df['BD'] = df.B * df.D
        df['CD'] = df.C * df.D
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])
        
    elif input_num == 5:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, \
                                    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, \
                                    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
                             'E' : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         )     
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                     1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['AD'] = df.A * df.D
        df['AE'] = df.A * df.E
        df['BC'] = df.B * df.C
        df['BD'] = df.B * df.D
        df['BE'] = df.B * df.E
        df['CD'] = df.C * df.D
        df['CE'] = df.C * df.E
        df['DE'] = df.D * df.E
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])
    
    B = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
    lst_B = B.T.tolist()
    
    equation = []
    for i in range(len(lst_B)):
        if i == 0:
            equation.append(str(round(lst_B[i], 4)))
        else:   
            equation.append('%s x %s' %(round(lst_B[i], 4), input_fcts[i]))
    
    str_equation = 'Response = ' + ' + '.join(equation)
    
    Y_mean = Y.T.mean()
    SS_tot = ((Y.T - Y_mean)**2).sum()
    lst_rsd = (Y - X @ B).T.tolist()
    lst_rsd_square = [rsd**2 for rsd in lst_rsd]
    SS_rsd = np.sum(lst_rsd_square)
    
    R2 = 'R2 = ' + str(round(1 - SS_rsd / SS_tot, 4))
    
    PRESS = 0
    for i in range(len(Y)):
        lst_rsd = (X[0:i+1, :] @ B - Y[0:i+1]).T.tolist()
        PRS = np.sum([rsd**2 for rsd in lst_rsd])
        PRESS += PRS
   
    Q2 = 'Q2 = ' + str(round(1 - PRESS / SS_tot, 4))
    
    if not n_clicks:
        return str_equation, R2, Q2, None
    
    elif n_clicks:
        return None, None, None, None

    
#  This callback predicts response based on inputs from sliders
@app.callback(
               Output('response', 'children'),
               Input('dd_num', 'value'),
               Input('dd_fcts', 'value'),
               Input('Y', 'value'),
               Input('slider_A', 'value'),
               Input('slider_B', 'value'),
               Input('slider_C', 'value'),
               Input('slider_D', 'value'),
               Input('slider_E', 'value')
             )
def update_prediction(input_num, input_fcts, input_Y, sl_A, sl_B, sl_C, sl_D, sl_E):
    if not input_num or not input_fcts:
        raise PreventUpdate
    
    mo = re.compile(r'\d*\.?\d+')
    Y = np.array(mo.findall(input_Y)).astype(float).T    # Get Y for matrix calculation
    
    if 2**input_num != len(Y):
        raise PreventUpdate
        
    if input_num == 3:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1]
                           }
                         )
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['BC'] = df.B * df.C
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])
        
        p_df = pd.DataFrame(
                             {
                               'A' : [sl_A],
                               'B' : [sl_B],
                               'C' : [sl_C]
                             }
                           )
        p_df['int'] = [1]
        p_df['AB'] = p_df.A * p_df.B
        p_df['AC'] = p_df.A * p_df.C
        p_df['BC'] = p_df.B * p_df.C
        p_X = np.array(p_df[input_fcts])
 
    elif input_num == 4:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         )
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['AD'] = df.A * df.D
        df['BC'] = df.B * df.C
        df['BD'] = df.B * df.D
        df['CD'] = df.C * df.D
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])

        p_df = pd.DataFrame(
                             {
                               'A' : [sl_A],
                               'B' : [sl_B],
                               'C' : [sl_C],
                               'D' : [sl_D]
                             }
                           )
        p_df['int'] = [1]
        p_df['AB'] = p_df.A * p_df.B
        p_df['AC'] = p_df.A * p_df.C
        p_df['AD'] = p_df.A * p_df.D
        p_df['BC'] = p_df.B * p_df.C
        p_df['BD'] = p_df.B * p_df.D
        p_df['CD'] = p_df.C * p_df.D
        p_X = np.array(p_df[input_fcts])        
        
    elif input_num == 5:
        df = pd.DataFrame(
                           {
                             'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, \
                                    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                             'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, \
                                    -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
                             'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
                             'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, \
                                    -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
                             'E' : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                           }
                         )     
        df['int'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                     1, 1, 1, 1, 1]
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['AD'] = df.A * df.D
        df['AE'] = df.A * df.E
        df['BC'] = df.B * df.C
        df['BD'] = df.B * df.D
        df['BE'] = df.B * df.E
        df['CD'] = df.C * df.D
        df['CE'] = df.C * df.E
        df['DE'] = df.D * df.E
        input_fcts.insert(0, 'int')
        X = np.array(df[input_fcts])
       
        p_df = pd.DataFrame(
                             {
                               'A' : [sl_A],
                               'B' : [sl_B],
                               'C' : [sl_C],
                               'D' : [sl_D],
                               'E' : [sl_E],
                             }
                           )
        p_df['int'] = [1]
        p_df['AB'] = p_df.A * p_df.B
        p_df['AC'] = p_df.A * p_df.C
        p_df['AD'] = p_df.A * p_df.D
        p_df['AE'] = p_df.A * p_df.E
        p_df['BC'] = p_df.B * p_df.C
        p_df['BD'] = p_df.B * p_df.D
        p_df['BE'] = p_df.B * p_df.E
        p_df['CD'] = p_df.C * p_df.D
        p_df['CE'] = p_df.C * p_df.E
        p_df['DE'] = p_df.D * p_df.E
        p_X = np.array(p_df[input_fcts])  
        
    B = np.linalg.inv((X.T @ X)) @ (X.T @ Y) 
    
    p_Y = p_X @ B
    
    [p_response] = p_Y.tolist() 
    
    str_p_response = 'Predicted response = %s' %round(p_response, 4)
    
    return str_p_response
    
    
if __name__ == '__main__':
    doe.run_server(debug = False, 
                   host="0.0.0.0",
                   port=port)
