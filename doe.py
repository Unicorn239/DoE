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
                    html.H2('DoE Matrix Analyzer',
                            style = {'textAlign' : 'center'}),
                    dbc.Row(
                        [
                          dbc.Col(
                              [
                                html.H3('Generate DoE design matrix'),
                                html.Br(),
                                html.Ul(
                                     [    
                                       html.Li('Number of variables selected determines the number of experiments \
                                                matrix rows'),
                                       html.Li('Experiments ought to be conducted as instructed with the matrix'),
                                       html.Li('Impact of interactions between variables could be assessed either \
                                                collectively or individually'),
                                       html.Li('Decision to either keep or remove variable(s) and/or interactions \
                                                depends on evaluation of R2/Q2') 
                                     ]
                                       ),
                                 dbc.Label('Select number of factors'),
                                 html.Br(),
                                 dcc.Dropdown(
                                               id = 'dd_num',
                                               options = [{'label' : '%s variables' %i, 'value' : i} \
                                                          for i in [3, 4, 5]],
                                               placeholder = 'Number of variables',
                                               style = {
                                                         'width' : '60%',
                                                         'height' : 25,
                                                         'margin-left' : '5%',
                                                         'background-color' : 'black',
                                                         'color' : 'black'
                                                       }    
                                             ),
                                  html.Br(),
                                  dbc.Label('Select factors for assessments'),
                                  html.Br(),
                                  dcc.Dropdown(
                                                id = 'dd_factor',
                                                placeholder = 'Factors for matrix calculation',
                                                multi = True,
                                                style = {
                                                          'width' : '60%',
                                                          'height' : 25,
                                                          'margin-left' : '5%',
                                                          'background-color' : 'black',
                                                          'color' : 'black'                                                        
                                                        }                                   
                                              ), 
                                  html.Br(),
                                  html.Br(),
                                  html.Br(),
                                  html.Br(),
                                  html.Br(),
                                  html.Br(),
                                  dbc.Label('Display design matrix'),
                                  html.Br(),
                                  html.Div(
                                            id = 'matrix',
                                            style = {
                                                      'width' : '80%',
                                                      'height' : 100,
                                                      'margin-left' : '5%'
                                                    }
                                          ),
                                  html.Br(),
                                  html.Br(),
                                  html.Button(
                                               'Clear',
                                               id = 'bt_clr',
                                               style = {                                                
                                                         'width' : '20%',
                                                         'height' : 25,
                                                         'background-color' : 'black',
                                                         'color' : 'white',
                                                         'margin-left' : '8.5%'                                                 
                                                        }
                                             ),
                              ],
                                md = 3,
                                lg = 6
                                 ),
                          
                          dbc.Col(
                              [
                                html.H3('Calculation and evaluation of regression coefficients'),
                                html.Br(),
                                html.Ul(
                                     [
                                         html.Li('Input values of response derived from conducting experiments \
                                                  instructed by DoE design matrix'),
                                         html.Li('Separate values either by space or comma'),
                                         html.Li('Regression equation, R2, Q2 will be displayed'),
                                         html.Li(
                                                  [
                                                    'Github homepage - ',
                                                    html.A('https://github.com/Unicorn239', 
                                                            href = 'https://github.com/Unicorn239')
                                                  ]
                                                 )
                                     ]
                                       ),
                                html.Br(),
                                html.Br(),
                                dbc.Label('Input \'response\' - Y'),
                                html.Br(),
                                dcc.Textarea(
                                              id = 'response',
                                              placeholder = 'Response - Y',
                                              style = {
                                                        'background-color' : 'black',
                                                        'color' : 'white',
                                                        'margin-left' : '5%',
                                                        'width' : '50%',
                                                        'height' : 50
                                                        }
                                              ),
                                html.Br(),
                                html.Br(),
                                dbc.Label('DoE model equation'),
                                html.Br(),
                                html.Div(
                                          id = 'equation',
                                          style = {
                                                    'width' : '80%',
                                                    'height' : 30,
                                                    'margin-left' : '5%'
                                                  }
                                        ),
                                html.Br(),
                                html.Br(),
                                html.Div(
                                          id = 'R_square',
                                          style = {
                                                    'width' : '80%',
                                                    'height' : 20,
                                                    'margin-left' : '5%'
                                                  }                                    
                                        ),
                                html.Br(),
                                html.Br(),
                                html.Div(
                                          id = 'Q_square',
                                          style = {
                                                    'width' : '80%',
                                                    'height' : 20,
                                                    'margin-left' : '5%'
                                                  }                                    
                                        ),
                                html.Br(),
                                html.Br(),
                                html.Button(
                                             'Clear',
                                             id = 'txt_clr',
                                             style = {
                                                       'width' : '20%',
                                                       'height' : 25,
                                                       'background-color' : 'black',
                                                       'color' : 'white',
                                                       'margin-left' : '8.5%'
                                                       }                                  
                                             )                                
                              ],
                                md = 3,
                                lg = 6
                                 )
                        ]
                           )
                  ]
                     )


@app.callback(
               Output('dd_factor', 'options'), 
               Input('dd_num', 'value')
             )
def update_dd_factor(input_num):  
    if not input_num:
        raise PreventUpdate
 
    if input_num == 3:
        return [{'label' : '%s' %factor, 'value' : factor} for factor in ['A', 'B', 'C', 'AB', 'AC', 'BC']]
    
    if input_num == 4:
        return [{'label' : '%s' %factor, 'value' : factor} for factor in ['A', 'B', 'C', 'D', 'AB', 'AC', 'AD', \
                                                                          'BC', 'BD', 'CD']]
    if input_num == 5:
        return [{'label' : '%s' %factor, 'value' : factor} for factor in ['A', 'B', 'C', 'D', 'E', 'AB', 'AC', \
                                                                 'AD', 'AE', 'BC', 'BD', 'BE', 'CD', 'CE', 'DE']]
                                                                


@app.callback(
               Output('dd_factor', 'value'),
               Output('matrix', 'children'),
               Output('bt_clr', 'n_clicks'),
               Input('dd_num', 'value'),
               Input('dd_factor', 'value'),
               Input('bt_clr', 'n_clicks')
             )
def update_matrix(input_num, input_factor, clr_clicks):
    if not input_num:
        raise PreventUpdate
    
    if input_num == 3:
        df = pd.DataFrame(
            {
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1]
            }
             )
        mx = np.matrix(df)
        
    elif input_num == 4:
        df = pd.DataFrame(
            {
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
              'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
             )
        mx = np.matrix(df)

    elif input_num == 5:
        df = pd.DataFrame(
            {
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
              'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
              'E' : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                     
            }
             )        
        mx = np.matrix(df)
        
    if not clr_clicks:
        return input_factor, str(mx), None
    else:
        return '', '', None


@app.callback(
               Output('equation', 'children'),
               Output('R_square', 'children'),
               Output('Q_square', 'children'),
               Output('txt_clr', 'n_clicks'),
               Input('dd_num', 'value'),
               Input('dd_factor', 'value'),    
               Input('response', 'value'),
               Input('txt_clr', 'n_clicks')
             )
def update_equation(input_num, input_factor, input_resp, txt_clicks):
    mo = re.compile(r'\d*\.?\d+')
    resp = pd.Series(mo.findall(input_resp)).astype(float)   
    
    if not input_num or not input_factor or 2**input_num != len(resp):
        raise PreventUpdate
    
    if input_num == 3:
        df = pd.DataFrame(
            {
              'intecept' : [1, 1, 1, 1, 1, 1, 1, 1],
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1]
            }
             )
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['BC'] = df.B * df.C
        
        input_factor.insert(0, 'intecept')
        X = np.matrix(df[input_factor])
        
    elif input_num == 4:
        df = pd.DataFrame(
            {
              'intecept' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
              'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
             )
        df['AB'] = df.A * df.B
        df['AC'] = df.A * df.C
        df['AD'] = df.A * df.D
        df['BC'] = df.B * df.C
        df['BD'] = df.B * df.D
        df['CD'] = df.C * df.D
        
        input_factor.insert(0, 'intecept')
        X = np.matrix(df[input_factor])

    elif input_num == 5:
        df = pd.DataFrame(
            {
              'intecept' : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              'A' : [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
              'B' : [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
              'C' : [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
              'D' : [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
              'E' : [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],                     
            }
             )
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
        
        input_factor.insert(0, 'intecept')
        X = np.matrix(df[input_factor]) 
        
    Y = np.matrix(resp).T
    B = (X.T * X).I * (X.T * Y)
    [lst_B] = B.T.tolist()
    lst_equation = []
    for i in range(len(lst_B)):
        if i == 0:
            lst_equation.append('%s'%(round(lst_B[i], 4))
        else:
            lst_equation.append('%s * %s'%(round(lst_B[i], 4), input_factor[i]))
    equation = 'Y = ' + ' + '.join(lst_equation)
    
    SS_tot = ((resp - resp.mean())**2).sum()
    mx_rsd = Y - X * B 
    [lst_rsd] = mx_rsd.T.tolist()
    se_rsd = pd.Series([i**2 for i in lst_rsd])
    SS_rsd = se_rsd.sum()
    R_square = round(1 - SS_rsd / SS_tot, 4)
    
# Calculation of Q2
    PRESS = 0
    for i in range(len(resp)):
        mx_rsd_q = Y - X * B
        [lst_rsd_q] = mx_rsd_q.T.tolist()
        PRESS_rsd = pd.Series([i**2 for i in lst_rsd_q]).sum()
        PRESS += PRESS_rsd
        Y = np.delete(Y, (0), axis = 0)
        X = np.delete(X, (0), axis = 0)

    Q_square = round(1 - PRESS / SS_tot, 4)
    
    if not txt_clicks:
        return equation, 'R^2 = %s'%R_square, 'Q^2 = %s'%Q_square, None
    
    else: 
        return '', '', '', None
    
    
if __name__ == '__main__':
    doe.run_server(debug = False, 
                   host="0.0.0.0",
                   port=port)
