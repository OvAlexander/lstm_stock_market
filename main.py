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
    
    # Sorting 
    df = df.sort_values('Date')
    df.head()

    # Ploting
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low']+df['High']/2.0)) # Plots the average (Mid Price)
    plt.xticks(range(0,df.shape[0],500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.show()

    # Data subset creation
    high_prices = df.loc[:, 'High'].to_numpy()
    low_prices = df.loc[:, 'Low'].to_numpy()
    mid_prices = (high_prices+low_prices)/2.0

    train_data = mid_prices[:11000]
    test_data = mid_prices[11000:]

    # Normalization
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    smoothing_window_size = 2500
    for di in range(0, 10000, smoothing_window_size):
        scaler.fit(train_data[di:di+smoothing_window_size,:])
        train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

    scaler.fit(train_data[di+smoothing_window_size:,:])
    train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)

    EMA = 0.0
    gamma = 0.1
    for ti in range(11000):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA
    
    all_mid_data = np.concatenate([train_data, test_data],axis=0)

    window_size = 100
    N = train_data.size
    std_avg_predicitons = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size, N):
        if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx,'Date']
        
        std_avg_predicitons.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_errors.append((std_avg_predicitons[-1]-train_data[pred_idx])**2)
        std_avg_x.append(date)
    print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

    # Ploting
    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True') # Plots the average (Mid Price)
    plt.plot(range(window_size, N), std_avg_predicitons, color='orange', label='Prediction') # Plots the MSE
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.show()