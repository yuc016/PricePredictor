import torch
from random import randint
import json
import os
import matplotlib.pyplot as plt

def get_config(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def load_experiment_state(net, optimizer, dir_path):
    path = os.path.join(dir_path, "experiment_state.pt")
    if os.path.isfile(path):
        print("Model state loaded!")
        experiment_state = torch.load(path)
        rand_seed = experiment_state['seed']
        epoch = experiment_state['epoch']
        net_state = experiment_state['net']
        optimizer_state = experiment_state['optimizer']

        net.load_state_dict(net_state)
        optimizer.load_state_dict(optimizer_state)

        # Put optimizer data onto CUDA core
        # Effect equals optimizer.cuda() although torch.optim doesn't have cuda() function
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        return rand_seed, epoch
    else:
        print("No saved model state found!")
        return randint(0, 1234567890), 0

def save_experiment_state(seed, epoch, net, optimizer, dir_path):
    net_state = net.state_dict()
    optimizer_state = optimizer.state_dict()
    experiment_state = {'seed': seed,
                        'epoch': epoch,
                        'net': net_state, 
                        'optimizer': optimizer_state}

    path = os.path.join(dir_path, "experiment_state.pt")
    torch.save(experiment_state, path)
    print("Model state saved!")

def log_stats(train_losses, val_losses, dir_path):
    make_plot([train_losses, val_losses], ["Training loss", "Validation loss"], 
              "Epoch", "Loss Curve (Last Training Session)", "loss_curve.png", 
              dir_path)


def make_plot(data, labels, x_axis_name, plot_name, file_name, dir_path):
    x = len(data[0])
    x_axis = [i for i in range(1, x+1)]

    for i in range(len(data)):
        plt.plot(x_axis, data[i], label=labels[i])
    plt.xlabel(x_axis_name)

    plt.legend()
    plt.title(plot_name)
    path = os.path.join(dir_path, file_name)
    plt.savefig(path)
    plt.close()