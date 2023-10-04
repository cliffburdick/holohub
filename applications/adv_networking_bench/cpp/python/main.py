import socket
import sys
import os
import struct
from scipy.io import loadmat
import scipy.signal as cusignal
import numpy as np
import numpy as cp
import matplotlib.pyplot as plt
import math
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from scipy.io import loadmat
from queue import Queue, Empty
import threading
from adi_nats import nats_async
from common_msg import subscribe, external_message
from loguru import logger
from dash.dependencies import Input, Output
import struct
import sys


L = 1228800
BW = 100
FFT_SIZE = 8192

np.set_printoptions(threshold=np.inf)


app = Dash(__name__)    

rxq_psd = Queue()
txq = Queue()
nats_inst = nats_async("10.110.102.191", {'spec_output':rxq_psd}, txq)
t = threading.Thread(target=nats_inst.start_async_loop, daemon=True)
t.start()    
txq.put_nowait(subscribe(subject="spec_output"))
data = pd.DataFrame({'x': range(1024), 'y': np.random.randn(1024)}) 
psd_fig = px.line(data, x="x", y="y")  


@app.callback(Output('PSD', 'figure'),
              Input('psd-interval', 'n_intervals'))
def update_psd(n):
    global psd_fig  
    try:
        obj = rxq_psd.get(block=False)    
        data_len = 1024 * 4
        #mtype, data = struct.unpack(f'@I{data_len}s', obj)
        (data,) = struct.unpack(f'@{data_len}s', obj)
        #assert(mtype == external_message.MSG_TYPE_PSD)
        x = np.frombuffer(data, dtype=np.float32)
        Fs = 2949120000/3
        l1 = 1024
        xlab = np.linspace(-Fs/2, Fs/2 - Fs/l1, l1)
        # data = pd.DataFrame({'x': W, 'y': x})    
        # psd_fig = px.line(data, x="x", y="y")            
        psd_fig.data[0]['x'] = xlab
        psd_fig.data[0]['y'] = x
        psd_fig.layout['uirevision'] = 'f'
        #print("Updating figure")
        return psd_fig
                 
    except Empty as e:
        return psd_fig

def main():

    # Get the figure for dummy plots
  

    
    app.layout = html.Div([
        html.Div(children=[   
            html.Div(children=[   
                html.Div(children=[   html.H2(children='Transmit Signal Configuration'),], style={'width':'100%','text-align': 'center'}),
            ]),
            html.Div(children=[ 
                html.Div(children=[html.Label(children='Center Frequency (MHz)'),html.Br(),html.Br(),html.Br(),html.Br(),html.Label(children='Transmit Gain (dB)')], style={'padding-right': '10px'}),
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 0')], style={'padding-right': '10px'}),
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 1')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 2')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 3')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 4')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 5')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 6')], style={'padding-right': '10px'}),   
                html.Div(children=[dcc.Input(value='', type='text', style={"width":'50%'}),dcc.Slider(min=-100, max=0, value=-40, vertical=True, verticalHeight=100),html.Label('Carrier 7')], style={'padding-right': '10px'}),
            ], style={'display':'flex', "background-color":"#B9E6A8", 'padding-top': '1%', 'padding-bottom': '1%'}),
            html.Br(),
            html.Div(children=[   
                html.Div(children=[   
                    html.Div(children=[   
                        html.Div(children=[   
                            html.Div(children=[ 
                                html.H2(children='Settings'),
                            ], style={'text-align': 'center'}),
                            html.Div(children=[   
                                html.Div(children=[   
                                    html.Label('TX RF CF'),
                                    dcc.Input(value='', type='text', style={"width":'50%'}),
                                ], style=dict(width='50%')),
                                html.Div(children=[   
                                    html.Label('RX RF CF'),
                                    dcc.Input(value='', type='text', style={"width":'50%'}),
                                ], style=dict(width='50%'))     
                            ], style=dict(width='100%',display='flex')),   
                        ],style={"background-color":"#E2E2E2"}),
                        html.Br(),
                        html.Div(children=[   
                            html.Div(children=[   
                                html.Label('TX Power dB', style={"width":'30%'}),
                                dcc.Input(value='', type='text', style={"width":'50%'}),
                            ], style=dict(width='50%')),
                            html.Div(children=[   
                                html.Label('Receive Mode'),
                                dcc.RadioItems(['One-Shot', 'Continuous'], 'One-Shot'),
                            ], style={"width":'38%', "background-color":"#B1B1B1"})     
                        ], style={"width":'100%',"display":'flex',"background-color":"#E2E2E2"}),
                        html.Br(),
                        html.Div(children=[   
                            html.Div(children=[   
                                html.Label('Transmitter'),
                            ], style=dict(width='50%')),
                            html.Div(children=[   
                                html.Label(children='Trigger'), html.Br(),html.Button('Start/Stop', id='trigger', n_clicks=0),
                                html.Br(),html.Br(),
                                html.Label('Digital Loopback'),
                                dcc.RadioItems(['Off', 'On'], 'Off'),
                            ], style=dict(width='50%'))     
                        ], style=dict(width='100%',display='flex'))                  

                    ], style={"width":'30%',"background-color":"#E2E2E2",'padding-left':'1%','padding-bottom':'1%'}),

                    html.Div(children=[   
                        html.H2(children='Measurements'),
                        html.Div(children=[ 
                            html.Label('Receive Digital Power dBfs'),
                            html.Br(),
                            dcc.Input(value='', type='text', style={"width":'20%'}),
                        ], style={"width":'50%'}),
                        html.Br(),
                        html.Div(children=[ 
                            html.Label('Carrier Select'),
                            dcc.Dropdown([0, 1, 2, 3, 4, 5, 6, 7], 0, id='carrier-select', style={"width":'50%'}),
                        ], style={"width":'50%'}),
                        html.Br(),                        
                        html.Div(id='evm_input', style={"width":'50%'})
                    ], style={"width":'70%', 'padding-left':'5%'})
                ], style=dict(display='flex'))
            ]),
        ], style={"width":'50%'}),
        html.Div(children=[  
            html.Div(children=[  
                html.Div(children=[   html.H2(children='Received Spectrum'),], style={'width':'100%','text-align': 'center'}),
                dcc.Graph(
                    id='PSD',
                    figure=psd_fig,
                ),
                dcc.Interval(
                    id='psd-interval',
                    interval=100, # in milliseconds
                    n_intervals=0
                ),
            ]),
            html.Div(children=[  
                html.Div(children=[html.Label(children='Measured Carrier Powers (dB)')], style={'padding-right': '3%'}),
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True, style={"width":'100%'}),html.Br(),html.Label('Carrier 0')], style={'padding-right': '3%'}),
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 1')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 2')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 3')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 4')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 5')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 6')], style={'padding-right': '3%'}),   
                html.Div(children=[dcc.Input(value='', type='text', readOnly=True,  style={"width":'100%'}),html.Br(),html.Label('Carrier 7')], style={'padding-right': '3%'}),                
            ], style=dict(display='flex')),
            html.Div(children=[ 
                html.Div(children=[   html.H2(children='Constellation Plot'),], style={'width':'100%','text-align': 'center'}),
                dcc.Graph(
                    id='constellation',
                    #figure=const_fig
                ),      
                dcc.Interval(
                    id='constfig-interval',
                    interval=100, # in milliseconds
                    n_intervals=0
                )                             
            ])  
        ], style=dict(width='50%'))
    ], style=dict(display='flex',width='60%')) 


    # t = threading.Thread(target=app.run, daemon=True, kwargs=dict(debug=True,host="192.168.1.133"))
    # t.start()
    logger.info("Starting Plotly server")    
    app.run(debug=True,host="10.110.102.191")


    # Start NATS thread

    

    iterations = 0

if __name__ == "__main__":
    main()