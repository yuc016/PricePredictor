import torch
import os
import json

def get_config(root_dir, file_name):
    path = os.path.join(root_dir, file_name)

    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)