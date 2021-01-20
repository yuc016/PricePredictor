import torch
from model import *
from dataset import *
from file_utils import *

class PPNeuralTrainer:
    def __init__(self, config_name):
        self.config = read_file_in_dir("./", config_name + ".json")

        self.net = PPNetV1(self.config["hidden_size"])
        self.criterion = None
        self.optimizer = torch.optim.Adam(self.net.parameters())

        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(self.config)

    def go():
        pass

    def train(self):
        self.net.train()

    def validate(self):
        self.net.eval()

    def test(self):
        self.net.eval()

