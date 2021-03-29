import requests
import time

import csv
import pandas as pd
import torch

def make_input(time_interval, input_size, output_size, encode_length, decode_length):
    raw_data = fetch_data(time_interval)
    return get_input_tensor(raw_data, time_interval, input_size, output_size, encode_length, decode_length)


def fetch_data(time_interval):
    raw_data = requests.get('https://api.pro.coinbase.com/products/BTC-USD/candles?granularity=' + str(time_interval), headers={'Cache-Control': 'no-cache'}).content.decode("utf-8") 
#     raw_data = requests.get('https://api.pro.coinbase.com/products/BTC-USD/candles?granularity=300', headers={'Cache-Control': 'no-cache'}).content.decode("utf-8") 
        
    raw_data = raw_data.replace('[[', '')
    raw_data = raw_data.replace(']]', '')

    raw_data = raw_data.split('],[')
    raw_data.append('time,low,high,open,close,volume')
    
    raw_data.reverse()
    
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i].split(',')
        
    return raw_data

def get_input_tensor(raw_data, time_interval, input_size, output_size, encode_length, decode_length):
    # Convert to panda dataframe
    time_series_df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
    
    # Assert that no timestep is missing
    time_stamps = time_series_df["time"]
    time_stamps = time_stamps.astype(float).values
    for i in range(len(time_stamps) - 1):
#         assert(time_stamps[i] + 300 == time_stamps[i+1])
        assert(time_stamps[i] + time_interval == time_stamps[i+1])
        
   # Make time serie tensor
    columns = ["time", "open", "high", "low", "close", "volume"]
    col_i = {}
    for i in range(len(columns)):
        col_i[columns[i]] = i 
    
    time_series_df = time_series_df[columns].astype(float)
    time_series_tensor = torch.tensor(time_series_df.values)
    
    # Keep to last full time bucket
    curr_time = time.time()
    if curr_time - time_interval < time_series_tensor[-1, col_i["time"]]:
        time_series_tensor = time_series_tensor[:-1]
    
    last_timestamp = time_series_tensor[-1, col_i["time"]]
    last_close = time_series_tensor[-1, col_i["close"]]
    
    
    closing_price_roc = (time_series_tensor[1:, col_i["close"]] - time_series_tensor[:-1, col_i["close"]]) / time_series_tensor[:-1, col_i["close"]] * 1000
    closing_price_roc = closing_price_roc.reshape(-1,1)
    
    candle_stats = (time_series_tensor[1:, col_i["open"]:col_i["volume"]] / time_series_tensor[:-1, col_i["close"]].reshape(-1,1) - 1) * 1000
    
    high_low_price = candle_stats[:, 1:3]
    
    # Summarize data by each TIME_INTERVAL, use Volume Weighted Average Price (VWAP)
#     step_size = time_interval // 300 # Number of data to aggregate into one model's timestep
#     shape = (time_series_tensor.shape[0] // step_size, input_size)
#     condensed_data_tensor = torch.empty(shape)

#     count = 0
#     j = 0
#     i = 0
#     while i <= time_series_tensor.shape[0] - step_size:
#         # Approximate trade price of all trades by averaging open and close price of the interval
#         avg = (time_series_tensor[i:i+step_size, col_i["open"]] + time_series_tensor[i:i+step_size, col_i["close"]]) / 2
        
#         # Calculate WVAP
#         volume_weighted_sum = torch.sum(time_series_tensor[i:i+step_size, col_i["volume"]] * avg)
#         total_volume = torch.sum(time_series_tensor[i:i+step_size, col_i["volume"]])
#         wvap = volume_weighted_sum / total_volume
        
#         # Assert that there is trade in the time interval
#         assert(total_volume != 0)
        
#         if input_size == 5:
#             condensed_data_tensor[j, col_i["open"]] = time_series_tensor[i, col_i["open"]]
#             condensed_data_tensor[j, col_i["low"]] = torch.min(time_series_tensor[i:i+step_size, col_i["low"]])
#             condensed_data_tensor[j, col_i["high"]] = torch.max(time_series_tensor[i:i+step_size, col_i["high"]])
#             condensed_data_tensor[j, col_i["close"]] = time_series_tensor[i+step_size-1, col_i["close"]]
            
#         condensed_data_tensor[j, -1] = wvap

#         i += step_size
#         j += 1

#     mini = torch.min(condensed_data_tensor).item()
#     maxi = torch.max(condensed_data_tensor).item()
#     mean = torch.mean(condensed_data_tensor).item()
#     print("Data summary: ")
#     print("\tMax ", maxi)
#     print("\tMin ", mini)
#     print("\tMean ", mean)
#     print()


    # Check for nans
    assert(torch.sum(closing_price_roc != closing_price_roc) == 0)
    assert(torch.sum(candle_stats != candle_stats) == 0)
    assert(torch.sum(high_low_price != high_low_price) == 0)
    
    data_in = candle_stats
    data_out = torch.cat([high_low_price, closing_price_roc], dim=1)
    
    assert(data_in.shape[1] == input_size)
    assert(data_out.shape[1] == output_size)

    # Make input time serie data and target time serie data tensor 
    num_data = (data_in.shape[0] - encode_length - decode_length) // decode_length + 1

    X_shape = (num_data, encode_length, input_size)
    y_shape = (num_data, decode_length, output_size)
    pred_serie_shape = (encode_length, input_size)

    X, y = torch.empty(X_shape), torch.empty(y_shape)
    pred_serie = torch.empty(pred_serie_shape)

    for i in range(num_data):
        start = i * decode_length
        X[i] = data_in[start:start+encode_length]
        y[i] = data_out[start+encode_length:start+encode_length+decode_length] # Price is last feature

    pred_serie[:, :] = data_in[-encode_length:]
    
    return X, y, pred_serie, last_timestamp, last_close
    