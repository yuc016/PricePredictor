import os
import sys
from datetime import datetime

import torch

import fileutils
import dataset
from trainer import NeuralNetTrainer

####
# Customizable data maker and data destination
from data_makers.dm_coinbase import make_input
DATA_FILE_PATH = "data/live/coinbase/data.csv"

# from data_makers.dm_cointelegraph import make_data
# DATA_FILE_PATH = "data/live/cointelegraph/amCharts.csv"
####


EXP_ROOT_DIR = "experiments"


if __name__ == "__main__":
    print()
    config_file_path = None
    experiment_dir_path = None

    # Must provide an experiment directory
    if len(sys.argv) < 3:
        raise Exception("Usage: python main.py <MODE> <experiment_name>\n" +
                        "\tMode: use t for full time serie model testing, p for single timestep predicting")

    mode = sys.argv[1]
    if mode not in ['t', 'p']:
        raise Exception("Mode not supported")
    
    experiment_name = sys.argv[2]
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)

    # Check experiment exists
    if not os.path.exists(experiment_dir_path):
        raise Exception(experiment_dir_path, " doesn't exist:")
    
    # Check config file exists
    config_file_path = os.path.join(experiment_dir_path, "config.json")
    if not os.path.isfile(config_file_path):
        raise Exception("config.json doesn't exist:")

    config = fileutils.get_config_from_file(config_file_path)
    
    
    # Prep data
    X, y, predict_serie, last_timestamp, last_close = make_input(config["data"]["time_interval"], 
                   config["model"]["input_feature_size"], 
                   config["model"]["output_feature_size"], 
                   config["data"]["input_serie_len"], 
                   config["data"]["output_serie_len"])

    # Prep network
    trainer = NeuralNetTrainer(config_file_path, experiment_dir_path)

    if mode == 't':
        dataloader = dataset.get_dataloader_from_tensor(config, X, y)
        trainer.test_dataloader = dataloader
        trainer.test("_test")
    elif mode == 'p':
        prediction = trainer.single_predict(predict_serie)
        print("AI prediction output:", prediction)
        print("Last timestamp:", str(int(last_timestamp)))
        print("\tIn PST:", datetime.utcfromtimestamp(last_timestamp - 25200).strftime('%Y-%m-%d %H:%M:%S'))
        print("Last close @ {:.3f} USD".format( last_close ))
        #print("Limit sell @ {:.3f} USD".format( last_close * (prediction[0] / 1000 + 1) ))
        #print("Limit buy @ {:.3f} USD".format( last_close * (prediction[1] / 1000 + 1) ))
        print("Predicted close: {:.3f} USD".format( last_close * (prediction[0] / 1000 + 1) ))
        