from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

def load_keys():
    load_dotenv()
    API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
    return API_KEY

if __name__ == '__main__':
    data_source = 'kaggle' # Possibly enum to kaggle or alphavantage
    api_key = load_keys()
    ticker = 'GOOG'
    url_str = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%{ticker}&outputsize=full&apikey=%{api_key}'
    save_file = f'stock_market_data-{ticker}.csv'

    if data_source == 'alpha':
        if not os.path.exists(save_file):
            with urllib.request.urlopen(url_str) as url:
                data = json.load(url.read().decode())
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
                for key, val in data.items():
                    date = dt.datetime.strptime(key, '%Y-%m-%d')
                    data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                float(v['4. close']), float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index += 1
            print(f'Data saved to : {save_file}')
            df.to_csv(save_file)
        
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(save_file)
    else:
        df = pd.read_csv(os.path.join('archive','Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date','Open','High','Low','Close']) # Looks at HPQ or HP Inc
        print('Loaded data from Kaggle repo')
    
    df = df.sort_values('Date')
    df.head()
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low']+df['High']/2.0)) # Plots the average (Mid Price)
    plt.xticks(range(0,df.shape[0],500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.show()