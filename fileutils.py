import torch
import json
import os
import matplotlib.pyplot as plt

def get_config(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def load_model_state(net, optimizer, dir_path):
    path = os.path.join(dir_path, "model_state.pt")
    if os.path.isfile(path):
        print("Model state loaded!")
        model_state = torch.load(path)
        net_state = model_state['net']
        optimizer_state = model_state['optimizer']

        net.load_state_dict(net_state)
        optimizer.load_state_dict(optimizer_state)

        # Put optimizer data onto CUDA core
        # Effect equals optimizer.cuda() although torch.optim doesn't have cuda() function
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        print("No saved model state found!")

def save_model_state(net, optimizer, dir_path):
    net_state = net.state_dict()
    optimizer_state = optimizer.state_dict()
    model_state = {'net': net_state, 'optimizer': optimizer_state}

    path = os.path.join(dir_path, "model_state.pt")
    torch.save(model_state, path)
    print("Model state saved!")

def log_stats(train_losses, val_losses, dir_path):
    e = len(train_losses)
    x_axis = [i for i in range(1, e+1)]
    plt.plot(x_axis, train_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.legend(loc='best')
    plt.title("Loss Curve (Last Training Session)")
    path = os.path.join(dir_path, "loss_curve.png")
    plt.savefig(path)