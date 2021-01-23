import torch
import copy
import fileutils
import model
import dataset

from math import sqrt


class PPNeuralTrainer:
    # Assume a valid config_file_path and model_state_file_path
    def __init__(self, config_file_path, experiment_dir_path):

        self.config = fileutils.get_config(config_file_path)
        self.experiment_dir_path = experiment_dir_path

        self.rand_seed, self.epoch = self.init_model(self.config)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataset.get_dataloaders(self.config, self.rand_seed)

    def init_model(self, config):
        input_size = self.config["model"]["input_size"]
        hidden_size = self.config["model"]["hidden_size"]
        num_lstm_layers = self.config["model"]["num_lstm_layers"]
        output_size = self.config["model"]["output_size"]
        learning_rate = self.config["training"]["learning_rate"]
        momentum = self.config["training"]["momentum"]

        self.net = model.PPNetV1(input_size, hidden_size, num_lstm_layers, output_size)
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Load saved model state if there is one, use same seed to get dataset
        rand_seed, epoch = fileutils.load_experiment_state(self.net, self.optimizer, self.experiment_dir_path)
        print("Random seed: ", rand_seed)
        print("Current epoch:", epoch)

        # Use GPU for training
        if torch.cuda.is_available():
            self.net = self.net.cuda().float()
            self.criterion = self.criterion.cuda()
        else:
            raise("CUDA Not Available, CPU training not implemented")

        return rand_seed, epoch

    def go(self):
        print("GO!")

        best_model = None
        min_val_loss = float("inf")
        train_losses = []
        val_losses = []
        num_epochs = self.config["training"]["num_epochs"]

        while self.epoch < num_epochs:
            train_loss = self.train()
            val_loss = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print("Validation loss at epoch ", self.epoch, ": ", val_loss)

            # Save best model
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(self.net)
                min_val_loss = val_loss

            self.epoch += 1

        # Save model state and log statistics
        fileutils.save_experiment_state(self.rand_seed, self.epoch, 
                                        self.net, self.optimizer, self.experiment_dir_path)
        fileutils.log_stats(train_losses, val_losses, self.experiment_dir_path)

        # if best_model is not None:
        #     self.net = best_model


    # Train an iteration through the training data in train_dataloader
    #   and optimize the neural net 
    # Return - averaged training loss
    def train(self):
        self.net.train()

        training_loss = 0
        
        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()

            # Make an extra dimension for input at each time step, which is 1 for PPV1
            X = X.unsqueeze(2)
            y = y.unsqueeze(2)

            predictions = self.net(X, y).squeeze(2)
            y = y.squeeze(2)
            loss = self.criterion(predictions, y)

            training_loss += loss.item()

            loss /= len(X) # Average over batch size
            loss.backward()

            self.optimizer.step()

        # Average by data points and data serie length, take square root to get RMSD from MSE
        return sqrt(training_loss / len(self.train_dataloader.dataset) / y.shape[1])


    # Run an iteration through the validation data in val_dataloader
    # Return - averaged validation loss
    def validate(self):
        self.net.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            for i, (X, y) in enumerate(self.val_dataloader):
                X, y = X.cuda(), y.cuda()

                # Make an extra dimension for input at each time step, which is 1 for PPV1
                X = torch.unsqueeze(X, 2)
                y = torch.unsqueeze(y, 2)

                predictions = self.net(X, y).squeeze(2)
                y = y.squeeze(2)
                loss = self.criterion(predictions, y)

                val_loss += loss.item()

        # Average by data points and data serie length, take square root to get RMSD from MSE
        return sqrt(val_loss / len(self.val_dataloader.dataset) / y.shape[1])


    def test(self):
        print("TEST!")
        self.net.eval()
        self.net.cpu()

        test_loss = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(self.test_dataloader):
                # Make an extra dimension for input at each time step, which is 1 for PPV1
                X = torch.unsqueeze(X, 2)
                y = torch.unsqueeze(y, 2)

                predictions = self.net(X, y).squeeze(2)
                y = y.squeeze(2)
                loss = self.criterion(predictions, y)

                if i == 0:
                    print(predictions * 100)
                    print(y * 100)

                test_loss += loss.item()

        # Average by data points and data serie length, take square root to get RMSD from MSE
        return sqrt(test_loss / len(self.test_dataloader.dataset) / y.shape[1])