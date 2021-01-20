import torch

from utils import *
from model import *
from dataset import *

class PPNeuralTrainer:
    def __init__(self, config_name):

        self.config = get_config("./", config_name + ".json")

        self.net = PPNetV1(self.config["model"]["hidden_size"])
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.net.parameters())

        if torch.cuda.is_available():
            self.net = self.net.cuda().float()
            self.criterion = self.criterion.cuda()
        else:
            raise("CUDA Not Available, CPU training not implemented")

        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloaders(self.config)

    def go(self):
        print("GO!")

        num_epochs = self.config["training"]["num_epochs"]

        train_losses = []
        val_losses = []

        for i in range(num_epochs):
            train_loss = self.train()
            val_loss = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    # Train an iteration through the training data in train_dataloader
    #   and optimize the neural net 
    # Return - averaged training loss
    def train(self):
        self.net.train()

        training_loss = 0
        
        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()

            predictions = self.net(X)
            loss = self.criterion(predictions, y)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Average loss over number of data and prediction serie length
        return training_loss / len(self.train_dataloader.dataset) / y.shape[1]

    # Run an iteration through the validation data in val_dataloader
    # Return - averaged validation loss
    def validate(self):
        self.net.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            for i, (X, y) in enumerate(self.val_dataloader):
                X, y = X.cuda(), y.cuda()

                predictions = self.net(X)
                loss = self.criterion(predictions, y)

                val_loss += loss.item()

        # Average loss over number of data and prediction serie length
        return val_loss / len(self.val_dataloader.dataset) / y.shape[1]

    def test(self):
        print("TEST!")
        self.net.eval()

