
import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import re
import string
import os
import json
import streamlit.components.v1 as components
from io import BytesIO
from time import sleep
import math

from numpy import *
import json
from pandas import DataFrame, Series
from numpy.random import randn
import io
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from streamlit_extras.app_logo import add_logo
import ccxt,pytz,time,schedule, requests
from datetime import datetime, timedelta, date
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from pprint import pprint
import statsmodels.regression.linear_model as rg


# Desiging & implementing changes to the standard streamlit UI/UX
st.set_page_config(page_icon="img/page_icon.png", layout='wide', initial_sidebar_state='expanded')    #Logo

st.title('Crypto Converter to Local Currency')
st.subheader("Navigate to side bar to see full project info as well as options to choose from, to get started!")

from forex_python.converter import CurrencyRates
from forex_python.converter import CurrencyCodes

from pylivecoinwatch import LiveCoinWatchAPI


YOUR_APP_ID = st.secrets["api_key"]

c = CurrencyRates()
    
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://github.com/chemicoPy/crypto-conv-dashboard/blob/main/img/page_icon.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

  
    # ------ layout setting---------------------------
st.sidebar.markdown(
            """
    ## Project Overview
    Crypto Converter is an app that converts crypto coin price to a local currency price. You can check price on a visualization board as well when you select the option and hit on the button.
    
    Get started already!""")    

  
st.sidebar.markdown("## Select Crypto pair & Interval below") # add a title to the sidebar container

instrument = st.sidebar.selectbox(
            "Convert From",
            ("BTC/USDT","ETH/USDT",
             "DOGE/USDT", "BNB/USDT", "USD/USDT", "XRP/USDT", "SOL/USDT", "TRX/USDT", "XAU/USDT", "LTC/USDT", "SHIB/USDT", "MATIC/USDT"),)

to_conv = st.sidebar.selectbox(
            "Convert To",
            ("GBP (British Pound Sterling)", 
             "EUR (Euro)", "NZD (New Zealand Dollar)", "USD (United States Dollar)", "NPR (Nepalese Rupee)", "JPY (Japanese Yen)","BGN (Bulgarian Lev)","CZK (Czech Republic Koruna)","DKK (Danish Krone)","HUF (Hungarian Forint)","PLN (Polish Zloty)","RON (Romanian Leu)","SEK (Swedish Krona)", 
                                                  "CHF (Swiss Franc)","ISK (Icelandic Króna)","NOK (Norwegian Krone)","TRY (Turkish Lira)","AUD (Australian Dollar)","BRL (Brazilian Real)","CAD (Canadian Dollar)","CNY (Chinese Yuan)","HKD (Hong Kong Dollar)","IDR (Indonesian Rupiah)","ILS (Israeli New Sheqel)", "INR (Indian Rupee)","KRW (South Korean Won)","MXN (Mexican Peso)","MYR (Malaysian Ringgit)","PHP (Philippine Peso)","SGD (Singapore Dollar)", "THB (Thai Baht)", "ZAR (South African Rand)", "NGN (Nigerian Naira)"),)   
Tframe = st.sidebar.selectbox(
        'Interval', ["Interval of interest", "1m","5m","15m","30m","1h","2h","4h","1d","1w","month"], index=0)


#Conversion

from pylivecoinwatch import LiveCoinWatchAPI

instrument_conv = instrument[:instrument.index("/")]
to_conv_2 = to_conv[:3]

res_2 = requests.get(
                "https://openexchangerates.org/api/latest.json",
                params = {
                    "app_id" : st.secrets["api_key"],
                    "symbols" : to_conv_2,
                    "show_alternatives": True
                        }
                )

rates_res_2 = res_2.json()["rates"]

lcw = LiveCoinWatchAPI()
lcw.set_api_key(st.secrets["api_key2"])

rate1 = lcw.coins_single(code=instrument_conv)["rate"]

conv_factor_1 = rate1
conv_factor_2 = rates_res_2[to_conv_2]

    
if st.sidebar.button("Show Viz!"):

  if instrument=="Select Forex Pair of interest" and Tframe=="Interval of interest":
    st.write("Kindly navigate to sidebar to see more options...")

  else:
    lim = 1000
    bybit = ccxt.bybit()
    instrument_conv = instrument[:instrument.index("/")]
    
    klines = bybit.fetch_ohlcv(instrument, timeframe=Tframe, limit= lim, since=None)

# filter klines to only include data from the past month
    from datetime import datetime, timedelta
    one_month_ago = datetime.now() - timedelta(days=1500)
    filtered_klines = [kline for kline in klines if datetime.fromtimestamp(kline[0]/1000) >= one_month_ago]

# convert the klines list to a DataFrame
    df = pd.DataFrame(filtered_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# convert the timestamp column to a datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# set the timestamp column as the index
    df.set_index('timestamp', inplace=True)
  
# Converting close price to local currency here

    for i in range(0, len(df['close'])):
        df['close'][i]= float(df['close'][i]) * (conv_factor_1) * (conv_factor_2)
    
    for i in range(0, len(df['open'])):
        df['open'][i]= float(df['open'][i]) * (conv_factor_1) * (conv_factor_2)
        
    for i in range(0, len(df['high'])):
        df['high'][i]= float(df['high'][i]) * (conv_factor_1) * (conv_factor_2)
    
    for i in range(0, len(df['low'])):
        df['low'][i]= float(df['low'][i]) * (conv_factor_1) * (conv_factor_2)

# calculate RSI
    df['rsi'] = ta.rsi(df['close'])

# calculate Stochastics
    stoch_data = df.copy()

#df['stoch_k'], df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close'])
#New calculation for Stochastic
    stoch_data['high'] = stoch_data['high'].rolling(14).max()
    stoch_data['low'] = stoch_data['low'].rolling(14).min()
    stoch_data['%k'] = (stoch_data["close"] - stoch_data['low'])*100/(stoch_data['high'] - stoch_data['low'])
    stoch_data['%d'] = stoch_data['%k'].rolling(3).mean()
    df['stoch_data'] = stoch_data['%d']


# Candlestick plot

    data = [go.Candlestick(x=df.index,
                       open=df.open,
                       high=df.high,
                       low=df.low,
                       close=df.close)]

    layout = go.Layout(title= f'{instrument} in {Tframe} Candlestick with Range Slider',
                   xaxis={'rangeslider':{'visible':True}})
    fig = go.Figure(data=data,layout=layout)
    plt.show()
    st.write(fig)

    
    import plotly.graph_objs as go

# create a line chart using the close data
    line = go.Scatter(
        x=df.index,
        y=df['close'],
        name='Close',
        line=dict(color='#17BECF')
)

# create the layout for the chart
    layout = go.Layout(
        title= f'{instrument} in {Tframe} Kline Data',
    xaxis=dict(
        title='Time',
        rangeslider=dict(visible=True)
    ),
    yaxis=dict(
        title='Price (USDT)'
    )
)

# create the figure and plot the chart
    fig = go.Figure(data=[line], layout=layout)
    plt.show()
    st.write(fig)

    import plotly.express as px

# create the line chart
    fig = px.line(df, x=df.index, y=df['rsi'], title=f'{instrument} in {Tframe} Kline Data')

# add the red and green lines
    fig.add_shape(
    type='line',
    x0=df.index[0],
    x1=df.index[-1],
    y0=75,
    y1=75,
    yref='y',
    line=dict(color='red', dash='dot')
)
        
    fig.add_shape(
    type='line',
    x0=df.index[0],
    x1=df.index[-1],
    y0=25,
    y1=25,
    yref='y',
    line=dict(color='green', dash='dot')
)

# set the y-axis range to include 0 and 100
    fig.update_layout(yaxis=dict(range=[0, 100]))

    plt.show()
    st.write(fig)

    st.write(df)
 

st.sidebar.markdown("## Quick Conversion") 
price = st.sidebar.number_input("Enter price to convert")
converted_price = float(price) * (conv_factor_1) * (conv_factor_2)

if st.sidebar.button("Convert"):
  st.sidebar.write("Converted Price = ", converted_price)
  
  
  
  
  
  
  
  
st.sidebar.markdown(

    """
    -----------
    # Other App(s):
 
    1. [Weather App](https://weather-monitor.streamlit.app/)
    
    """)
    
    
st.sidebar.markdown(

    """
    -----------
    # Let's connect
 
    [![Victor Ogunjobi](https://img.shields.io/badge/Author-@VictorOgunjobi-gray.svg?colorA=gray&colorB=dodgergreen&logo=github)](https://www.github.com/chemicopy)
    [![Victor Ogunjobi](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white)](https://www.linkedin.com/in/victor-ogunjobi-a761561a5/)
    [![Victor Ogunjobi](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=gray)](https://twitter.com/chemicopy_)
    """)



   
