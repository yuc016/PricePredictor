# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import json
import random
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import csv


# %%
# Dataset config
with open("../../config/config.json") as json_file:
    config = json.load(json_file)
    TIME_BEG = config["data"]["begin_timestamp"]
    # TIME_BEG = 1451606420 # = year 2016
    # TIME_BEG = 1483257600 # = year 2017
    # TIME_BEG = 1546347600 # = year 2019
    TIME_INTERVAL = config["data"]["time_interval"] // 60 # minutes
    INPUT_SERIE_LEN = config["data"]["input_serie_len"]
    OUTPUT_SERIE_LEN = config["data"]["output_serie_len"]
    INPUT_FEATURE_SIZE = config["model"]["input_feature_size"]
    OUTPUT_FEATURE_SIZE = config["model"]["output_feature_size"]
    
    print("Time interval: ", TIME_INTERVAL)
    print("Input serie length: ", INPUT_SERIE_LEN)
    print("Output serie length: ", OUTPUT_SERIE_LEN)
    print("Input feature size: ", INPUT_FEATURE_SIZE)
    print("Output feature size: ", OUTPUT_FEATURE_SIZE)
    


# %%
# Read csv file
prices = None
with open('kaggle_1min.csv', 'r') as f:
    reader = csv.reader(f)
    prices = list(reader)


# %%
# Process with Panda dataframe
prices_df = pd.DataFrame(prices[1:], columns=prices[0])
print("Original dataframe: \n", prices_df)

prices_df = prices_df.drop(columns=["Volume_(Currency)"])
prices_df = prices_df.astype(float)
prices_df = prices_df.fillna(0) # Replace NaN with 0 (time where trade volume = 0)
prices_df = prices_df.loc[prices_df["Timestamp"] >= TIME_BEG] # Cut off from desired begin time
prices_df = prices_df.reset_index(drop=True)

print("Processed dataframe: \n", prices_df)


# %%
# Make col name to index map
cols = prices_df.columns
col_i = dict()
for i in range(len(cols)):
    col_i[cols[i]] = i
print(col_i)


# %%
# Convert dataframe to torch tensor
prices_tensor = torch.from_numpy(prices_df.values)


# %%
# Fill values for timesteps with 0 trade volume
count = 0
for i in range(0, len(prices_tensor)):
    if prices_tensor[i, col_i["Volume_(BTC)"]] == 0 or prices_tensor[i, col_i["Low"]] < 200: # Fix a bad data with low price 1.5
        prices_tensor[i, col_i["Open"]] = prices_tensor[i-1, col_i["Close"]]
        prices_tensor[i, col_i["Close"]] = prices_tensor[i-1, col_i["Close"]]
        prices_tensor[i, col_i["High"]] = prices_tensor[i-1, col_i["Close"]]
        prices_tensor[i, col_i["Low"]] = prices_tensor[i-1, col_i["Close"]]
        prices_tensor[i, col_i["Weighted_Price"]] = prices_tensor[i-1, col_i["Close"]]
        count += 1
        
print("Fixed", str(count), "bad data out of ", str(len(prices_tensor)))


# %%
# Condense data by each TIME_INTERVAL, use Volume Weighted Average Price (VWAP)
num_data = prices_tensor.shape[0] // TIME_INTERVAL
shape = (num_data, len(col_i))
condensed_prices_tensor = torch.empty(shape)

count = 0
i = 0
j = 0
while i <= len(prices_tensor) - TIME_INTERVAL:
    condensed_prices_tensor[j, col_i["Timestamp"]] = prices_tensor[i, col_i["Timestamp"]] # Time stamp

    # Calculate WVAP
    volume_weighted_sum = torch.sum(prices_tensor[i:i+TIME_INTERVAL, col_i["Volume_(BTC)"]] * prices_tensor[i:i+TIME_INTERVAL, col_i["Weighted_Price"]])
    total_volume = torch.sum(prices_tensor[i:i+TIME_INTERVAL, col_i["Volume_(BTC)"]])
    wvap = volume_weighted_sum / total_volume
    
    # If no trade in this time interval use the price in the previous one
    # Here assume that first TIME_INTERVAL starting from TIME_BEG has non-zero trade volume
    if total_volume == 0:
        condensed_prices_tensor[j, col_i["Weighted_Price"]] = condensed_prices_tensor[j-1, col_i["Weighted_Price"]]
        count += 1
    else:
        condensed_prices_tensor[j, col_i["Weighted_Price"]] = wvap
        
    # Sum up candle stats
    condensed_prices_tensor[j, col_i["Open"]] = prices_tensor[i, col_i["Open"]] # Open price 
    condensed_prices_tensor[j, col_i["High"]] = torch.max(prices_tensor[i:i+TIME_INTERVAL, col_i["High"]]) # Highest price
    condensed_prices_tensor[j, col_i["Close"]] = prices_tensor[i+TIME_INTERVAL-1, col_i["Close"]] # Close price
    condensed_prices_tensor[j, col_i["Low"]] = torch.min(prices_tensor[i:i+TIME_INTERVAL, col_i["Low"]]) # Lowest price
    condensed_prices_tensor[j, col_i["Volume_(BTC)"]] = total_volume

    i += TIME_INTERVAL
    j += 1

print("Number of 0 volume time steps: ", str(count), "out of", str(len(condensed_prices_tensor)))


# %%
# Check low high price validity
for i in range(len(condensed_prices_tensor)):
    assert(condensed_prices_tensor[i, col_i["Low"]] == torch.min(condensed_prices_tensor[i, 1:5]))
    assert(condensed_prices_tensor[i, col_i["High"]] == torch.max(condensed_prices_tensor[i, 1:5]))


# %%
print(condensed_prices_tensor.shape)
torch.sum((condensed_prices_tensor[:, col_i["High"]] > condensed_prices_tensor[:, col_i["Open"]]) * (condensed_prices_tensor[:, col_i["Low"]] < condensed_prices_tensor[:, col_i["Close"]]))


# %%
# Data summary and visualization
ZOOM_IN_SECTION_LEN = 2000

print("Original price data shape: ", prices_tensor.shape)
print("Time condensed price data shape: ", condensed_prices_tensor.shape)

for col_name in col_i:
    ind = col_i[col_name]
    x = condensed_prices_tensor[:, 0] # Time
    y = condensed_prices_tensor[:, ind]
    
    # Random section for zoomed in view
    rand_sec_i = random.randint(0, len(condensed_prices_tensor) - ZOOM_IN_SECTION_LEN)

    fig, axs = plt.subplots(2)
    fig.suptitle(col_name)
    axs[0].plot(x, y)
    axs[1].plot(x[rand_sec_i:rand_sec_i+ZOOM_IN_SECTION_LEN], y[rand_sec_i:rand_sec_i+ZOOM_IN_SECTION_LEN])

    mini = torch.min(condensed_prices_tensor[:,ind]).item()
    maxi = torch.max(condensed_prices_tensor[:,ind]).item()
    mean = torch.mean(condensed_prices_tensor[:,ind]).item()
    std = torch.std(condensed_prices_tensor[:,ind]).item()

    print(col_name)
    print("Max", col_name, maxi)
    print("Min", col_name, mini)
    print("Mean", col_name, mean)
    print("Standard deviation of", col_name, std)
    print()


# %%
# # Visualize sectional price data
# x = condensed_prices_tensor[126000:, 0] # Time
# y = condensed_prices_tensor[126000:, col_i["Weighted_Price"]]

# fig, axs = plt.subplots(1)
# fig.suptitle(col_name)
# axs.plot(x, y)


# %%
# weighted price -> weighted price rate of change                  
weighted_roc = (condensed_prices_tensor[1:, col_i["Weighted_Price"]] - condensed_prices_tensor[:-1, col_i["Weighted_Price"]]) / condensed_prices_tensor[:-1, col_i["Weighted_Price"]]
weighted_roc *= 1000

# Checking for nan data
assert(torch.sum(weighted_roc != weighted_roc) == 0)

mini = torch.min(weighted_roc).item()
maxi = torch.max(weighted_roc).item()
mean = torch.mean(weighted_roc).item()
std = torch.std(weighted_roc).item()
percentile_lowest = np.percentile(weighted_roc.numpy(), 20).item()
percentile_lower = np.percentile(weighted_roc.numpy(), 40).item()
percentile_higher = np.percentile(weighted_roc.numpy(), 60).item()
percentile_highest = np.percentile(weighted_roc.numpy(), 80).item()

print("Max", maxi)
print("Min", mini)
print("Mean", mean)
print("Standard deviation", std)
print("percentile_lowest", percentile_lowest)
print("percentile_lower", percentile_lower)
print("percentile_higher", percentile_higher)
print("percentile_highest", percentile_highest)

time_x = condensed_prices_tensor[1:, 0]
price_change_y = weighted_roc
section_start = random.randint(0, len(weighted_roc) - ZOOM_IN_SECTION_LEN)

fig, axs = plt.subplots(2)
fig.suptitle('Change')
axs[0].plot(time_x, price_change_y)
axs[1].plot(time_x[section_start:section_start+ZOOM_IN_SECTION_LEN], price_change_y[section_start:section_start+ZOOM_IN_SECTION_LEN])

weighted_roc = weighted_roc.reshape(-1, 1)


# %%
# Convert candle stats [open high low close] to percentage of last closing price
candle_stats = condensed_prices_tensor[1:, 1:5] / condensed_prices_tensor[:-1, col_i["Close"]].reshape(-1, 1) - 1
candle_stats *= 1000
assert(torch.sum(candle_stats != candle_stats) == 0)


# %%
print(torch.max(candle_stats[:, 0]))
print(torch.min(candle_stats[:, 0]))
print(torch.max(candle_stats[:, 1]))
print(torch.min(candle_stats[:, 1]))
print(torch.max(candle_stats[:, 2]))
print(torch.min(candle_stats[:, 2]))


# %%
high_low_stats = candle_stats[:,1:3]


# %%
# Convert closing price to ROC
closing_price_roc = candle_stats[:, -1]
# closing_price_roc = (condensed_prices_tensor[1:, col_i["Close"]] - condensed_prices_tensor[:-1, col_i["Close"]]) / condensed_prices_tensor[:-1, col_i["Close"]] * 1000
closing_price_roc = closing_price_roc.reshape(-1,1)
assert(torch.sum(closing_price_roc != closing_price_roc) == 0)


# %%
# Normalize volume count
volumes = condensed_prices_tensor[1:, col_i["Volume_(BTC)"]]
mean_vol = torch.mean(volumes)
std_vol = torch.std(volumes)
volumes = (volumes - mean_vol) / std_vol
volumes = volumes.reshape(-1, 1)


# %%
# Slap all features together!
# data_in = weighted_roc
data_in = candle_stats
# data_in = torch.cat([candle_stats, closing_price_roc], dim=1)
    
# data_out = high_low_stats
data_out = closing_price_roc
# data_in = candle_stats
# data_out = torch.cat([high_low_stats, closing_price_roc], dim=1)

print(condensed_prices_tensor[:5])
print(data_in[:5])
print(data_out[:5])

assert(data_in.shape[0] == data_out.shape[0])
assert(data_in.shape[1] == INPUT_FEATURE_SIZE)
assert(data_out.shape[1] == OUTPUT_FEATURE_SIZE)


# %%
# Make input time serie data and target time serie data tensor 
num_data = (data_in.shape[0] - INPUT_SERIE_LEN - OUTPUT_SERIE_LEN) // OUTPUT_SERIE_LEN + 1

X_shape = (num_data, INPUT_SERIE_LEN, INPUT_FEATURE_SIZE)
y_shape = (num_data, OUTPUT_SERIE_LEN, OUTPUT_FEATURE_SIZE)

X, y = torch.empty(X_shape), torch.empty(y_shape)

for i in range(num_data):
    start = i * OUTPUT_SERIE_LEN
    X[i] = data_in[start:start+INPUT_SERIE_LEN]
    y[i] = data_out[start+INPUT_SERIE_LEN:start+INPUT_SERIE_LEN+OUTPUT_SERIE_LEN] # Price is last feature


# %%
print("Encode time serie data shape:", X_shape)
print("Decode time serie data shape:", y_shape)
# print("Condensed data:\n", condensed_prices_tensor[:INPUT_SERIE_LEN+OUTPUT_SERIE_LEN+1])
print("Condensed data:\n", condensed_prices_tensor[:2])
print("Training X:\n", X[:3])
print("Training y:\n", y[:4])


# %%
torch.save(X, "./../X.pt")
torch.save(y, "./../y.pt")


# %%



