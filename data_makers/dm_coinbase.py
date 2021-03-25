import requests
from datetime import datetime

import csv
import pandas as pd
import torch


def make_data(time_interval, encode_length, decode_length, data_file_path):
    fetch_data(time_interval, data_file_path)
    return get_xy_tensors(time_interval, encode_length, decode_length, data_file_path)


def fetch_data(time_interval, data_file_path):
    # There is no 1800 granularity so use 900 then trim
    if time_interval == 1800:
        time_interval = 900

    raw_data = requests.get('https://api.pro.coinbase.com/products/BTC-USD/candles?granularity=' + str(time_interval)).content.decode("utf-8") 

    raw_data = raw_data.replace('[[', '')
    raw_data = raw_data.replace(']]', '')

    raw_data = raw_data.split('],[')
    raw_data.reverse()

    raw_data = ''.join(x + "\n" for x in raw_data)

    with open(data_file_path, "w") as file:
        file.write('"time","low","high","open","close","volume"\n')
        file.write(raw_data)


def get_xy_tensors(time_interval, encode_length, decode_length, data_file_path):
    column = "close"

    # Read csv file and parse data into tensor
    time_series = None
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        time_series = list(reader)
    
    time_series_df = pd.DataFrame(time_series[1:], columns=time_series[0])

    # Assert that no data is missing
    time_stamps = time_series_df["time"]
    time_stamps = time_stamps.astype(float)
    for i in range(len(time_stamps) - 1):
        jump = 900 if time_interval == 1800 else time_interval
        assert(time_stamps[i] + jump == time_stamps[i+1])
    # UTC - 8hr = PST
    print("Time of latest data point: ", datetime.utcfromtimestamp(time_stamps[i+1] - 25200).strftime('%Y-%m-%d %H:%M:%S'))
    print()

    time_series_df = time_series_df[[column]]
    time_series_df = time_series_df.astype(float)
    time_series_tensor = torch.tensor(time_series_df[column].values)
    
    # Get every other to obtain 30 mins time interval data
    if time_interval == 1800:
        time_series_tensor = time_series_tensor[torch.arange(1, len(time_series_tensor), 2)]

    mini = torch.min(time_series_tensor).item()
    maxi = torch.max(time_series_tensor).item()
    mean = torch.mean(time_series_tensor).item()

    print("Data summary: ")
    print("\tMax ", maxi)
    print("\tMin ", mini)
    print("\tMean ", mean)
    print()

    # Convert series data to change in percentage in one thousanth
    series_rate_change = (time_series_tensor[1:] - time_series_tensor[:-1]) / time_series_tensor[:-1]
    series_rate_change *= 1000

    mini = torch.min(series_rate_change).item()
    maxi = torch.max(series_rate_change).item()
    mean = torch.mean(series_rate_change).item()

    print("Rate of change (in 1000th):")
    print("\tMax", maxi)
    print("\tMin", mini)
    print("\tMean", mean)
    print()

    # Check for nans
    assert(torch.sum(series_rate_change != series_rate_change) == 0)

    data = series_rate_change.reshape(-1, 1)

    # Make input time serie data and target time serie data tensor 
    num_data = (len(data) - encode_length - decode_length) // decode_length + 1

    X_shape = (num_data, encode_length, len(data[0]))
    y_shape = (num_data, decode_length)

    X, y = torch.empty(X_shape), torch.empty(y_shape)

    for i in range(num_data):
        start = i * decode_length
        X[i] = data[start:start+encode_length]
        y[i] = data[start+encode_length:start+encode_length+decode_length, -1] # Price is last feature

    print("Number of data: ", str(len(X)))
    print()

    return X, y