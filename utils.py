import torch
import os
import json
import matplotlib.pyplot as plt

def get_config(root_dir, file_name):
    path = os.path.join(root_dir, file_name)

    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)

def save_model(model, root_dir, config_name):
    path = os.path.join(root_dir, config_name + ".pt")
    torch.save(model.state_dict(), path)

def log_stats(train_losses, val_losses, root_dir, config_name):
    e = len(train_losses)
    x_axis = [i for i in range(1, e+1)]
    plt.figure()
    plt.plot(x_axis, train_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.title(config_name + " loss curve")
    path = os.path.join(root_dir, config_name + "_loss_curve.png")
    plt.savefig(path)
    plt.show()