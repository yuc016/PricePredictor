import requests
from datetime import datetime

import csv
import pandas as pd
import torch

NEWEST = False

def make_input(time_interval, encode_length, decode_length, data_file_path):
    fetch_data(time_interval, data_file_path)
    return get_input_tensor(time_interval, encode_length, decode_length, data_file_path)


def fetch_data(time_interval, data_file_path):
    raw_data = requests.get('https://api.pro.coinbase.com/products/BTC-USD/candles?granularity=300').content.decode("utf-8") 
        
    raw_data = raw_data.replace('[[', '')
    raw_data = raw_data.replace(']]', '')

    raw_data = raw_data.split('],[')
    raw_data.reverse()

    raw_data = ''.join(x + "\n" for x in raw_data)

    with open(data_file_path, "w") as file:
        file.write('"time","low","high","open","close","volume"\n')
        file.write(raw_data)


def get_input_tensor(time_interval, encode_length, decode_length, data_file_path):
    # Read csv file
    time_series = None
    with open(data_file_path, 'r') as f:
        time_series = list(csv.reader(f))

    # Convert to panda dataframe
    time_series_df = pd.DataFrame(time_series[1:], columns=time_series[0])

    # Assert that no timestep is missing
    time_stamps = time_series_df["time"]
    time_stamps = time_stamps.astype(float).values
    for i in range(len(time_stamps) - 1):
        assert(time_stamps[i] + 300 == time_stamps[i+1])
        
    # Print latest time (UTC - 8hr = PST)
    print("Time of latest data point: ", datetime.utcfromtimestamp(time_stamps[i+1] - 25200).strftime('%Y-%m-%d %H:%M:%S'))
        
    # Make time serie tensor
    time_series_df = time_series_df[["open", "close", "volume"]].astype(float)
    time_series_tensor = torch.tensor(time_series_df.values)
    
    # Summarize data by each TIME_INTERVAL, use Volume Weighted Average Price (VWAP)
    step_size = time_interval // 300 # Number of data to aggregate into one model's timestep
    shape = (time_series_tensor.shape[0] // step_size, 1)
    condensed_data_tensor = torch.empty(shape)

    count = 0
    j = 0
    i = 0
    while i <= time_series_tensor.shape[0] - step_size:
        # Approximate trade price of all trades by averaging open and close price of the interval
        avg = torch.mean(time_series_tensor[i:i+step_size, :2], dim=1)
        
        # Calculate WVAP
        volume_weighted_sum = torch.sum(time_series_tensor[i:i+step_size, 2] * avg)
        total_volume = torch.sum(time_series_tensor[i:i+step_size, 2])
        wvap = volume_weighted_sum / total_volume
        
        # Assert that there is trade in the time interval
        assert(total_volume != 0)

        condensed_data_tensor[j, 0] = wvap

        i += step_size
        j += 1

    print("Latest vwap", condensed_data_tensor[-1].item(), "USD\n")
        
#     mini = torch.min(condensed_data_tensor).item()
#     maxi = torch.max(condensed_data_tensor).item()
#     mean = torch.mean(condensed_data_tensor).item()
#     print("Data summary: ")
#     print("\tMax ", maxi)
#     print("\tMin ", mini)
#     print("\tMean ", mean)
#     print()

    # Convert series data to change in percentage in one thousanth
    series_rate_change = (condensed_data_tensor[1:, 0] - condensed_data_tensor[:-1, 0]) / condensed_data_tensor[:-1, 0]
    series_rate_change *= 1000

#     mini = torch.min(series_rate_change).item()
#     maxi = torch.max(series_rate_change).item()
#     mean = torch.mean(series_rate_change).item()

#     print("Rate of change (in 1000th):")
#     print("\tMax", maxi)
#     print("\tMin", mini)
#     print("\tMean", mean)
#     print()

    # Check for nans
    assert(torch.sum(series_rate_change != series_rate_change) == 0)

    X = series_rate_change.reshape(-1, 1)

    return X