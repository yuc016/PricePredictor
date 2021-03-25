import torch
from random import randint
import json
import os
import matplotlib.pyplot as plt

def get_config_from_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def load_experiment_state(trainer):
    path = os.path.join(trainer.experiment_dir_path, "experiment_state.pt")
    if os.path.isfile(path):        
        experiment_state = torch.load(path)
        trainer.rand_seed = experiment_state['seed']
        trainer.epoch = experiment_state['epoch']
        trainer.best_score = experiment_state['best_score']
        trainer.train_losses = experiment_state['train_losses']
        trainer.val_losses = experiment_state['val_losses']
        net_state = experiment_state['net']
        best_net_state = experiment_state['best_net']
        optimizer_state = experiment_state['optimizer']

        trainer.net.load_state_dict(net_state)
        trainer.best_net.load_state_dict(best_net_state)
        trainer.optimizer.load_state_dict(optimizer_state)

        # Put optimizer data onto CUDA core
        # Effect equals optimizer.cuda() although torch.optim doesn't have cuda() function
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        print("Experiment state loaded!")
    else:
        print("No saved model state found!")
        trainer.rand_seed = randint(0, 1234567890)
        trainer.epoch = 0


def save_experiment_state(trainer):
    net_state = trainer.net.state_dict()
    best_net_state = trainer.best_net.state_dict()
    optimizer_state = trainer.optimizer.state_dict()
    
    experiment_state = {'seed': trainer.rand_seed,
                        'epoch': trainer.epoch,
                        'net': net_state,
                        'best_net': best_net_state,
                        'optimizer': optimizer_state,
                        'best_score': trainer.best_score,
                        'train_losses': trainer.train_losses,
                        'val_losses': trainer.val_losses
                        }

    path = os.path.join(trainer.experiment_dir_path, "experiment_state.pt")
    torch.save(experiment_state, path)
    print("Experiment state saved!")


def log_loss_stats(train_losses, val_losses, dir_path):
    make_plot([train_losses, val_losses], ["Training loss", "Validation loss"], 
              "Epoch", "Loss Curve", "loss_curve.png", 
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