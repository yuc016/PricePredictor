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
    e = len(train_losses)
    x_axis = [i for i in range(1, e+1)]
    plt.plot(x_axis, train_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Loss Curve (Last Training Session)")
    path = os.path.join(dir_path, "loss_curve.png")
    plt.savefig(path)