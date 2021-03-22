import csv
import pandas as pd
import torch

import os
import sys

import fileutils
import dataset
from trainer import NeuralNetTrainer


EXP_ROOT_DIR = "./experiment"
DATA_ROOT_DIR = "./data/live"
DATA_FILE_NAME = "amCharts.csv"


def get_data_tensor(config):

    data_file_path = os.path.join(DATA_ROOT_DIR, DATA_FILE_NAME)

    # Read csv file and parse data into tensor
    time_series = None
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        time_series = list(reader)
    
    time_series_df = pd.DataFrame(time_series[1:], columns=time_series[0])
    time_series_df = time_series_df[["weighted"]]
    time_series_df = time_series_df.astype(float)

    time_series_tensor = torch.tensor(time_series_df["weighted"].values)

    mini = torch.min(time_series_tensor).item()
    maxi = torch.max(time_series_tensor).item()
    mean = torch.mean(time_series_tensor).item()
    std = torch.std(time_series_tensor).item()

    print("Data summary: ")
    print("\tMax ", maxi)
    print("\tMin ", mini)
    print("\tMean ", mean)
    print("\tStandard deviation", std)
    print()

    # Convert series data to change in percentage in one thousanth
    series_rate_change = (time_series_tensor[1:] - time_series_tensor[:-1]) / time_series_tensor[:-1]
    series_rate_change *= 1000

    mini = torch.min(series_rate_change).item()
    maxi = torch.max(series_rate_change).item()
    mean = torch.mean(series_rate_change).item()
    std = torch.std(series_rate_change).item()

    print("Converted to rate of change (in 1000th) summary:")
    print("\tMax", maxi)
    print("\tMin", mini)
    print("\tMean", mean)
    print("\tStandard deviation", std)

    # Check for nans
    assert(torch.sum(series_rate_change != series_rate_change) == 0)

    series_rate_change = series_rate_change.reshape(-1, 1)

    data = series_rate_change

    LEN_ENCODE_SERIE = config["model"]["len_encode_serie"]
    LEN_DECODE_SERIE = config["model"]["len_decode_serie"]

    # Make input time serie data and target time serie data tensor 
    num_data = (len(data) - LEN_ENCODE_SERIE - LEN_DECODE_SERIE) // LEN_DECODE_SERIE + 1

    X_shape = (num_data, LEN_ENCODE_SERIE, len(data[0]))
    y_shape = (num_data, LEN_DECODE_SERIE)

    X, y = torch.empty(X_shape), torch.empty(y_shape)

    for i in range(num_data):
        start = i * LEN_DECODE_SERIE
        X[i] = data[start:start+LEN_ENCODE_SERIE]
        y[i] = data[start+LEN_ENCODE_SERIE:start+LEN_ENCODE_SERIE+LEN_DECODE_SERIE, -1] # Price is last feature

    return X, y

if __name__ == "__main__":
    config_file_path = None
    experiment_dir_path = None

    # Must provide an experiment directory
    if len(sys.argv) < 2:
        raise Exception("Usage: python main.py <experiment_name>")

    experiment_name = sys.argv[1]
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)

    # Check experiment exists
    if not os.path.exists(experiment_dir_path):
        raise Exception(experiment_dir_path, " doesn't exist:")
    
    # Check config file exists
    config_name = sys.argv[1]
    config_file_path = os.path.join(experiment_dir_path, "config")
    if not os.path.isfile(config_file_path):
        raise Exception("config doesn't exist:")

    config = fileutils.get_config(config_file_path)

    # Prep data
    X, y = get_data_tensor(config)
    dataloader = dataset.get_dataloader_from_tensor(config, X, y)

    # Prep network
    trainer = NeuralNetTrainer(config_file_path, experiment_dir_path)
    trainer.test_dataloader = dataloader
    trainer.test("_predict")