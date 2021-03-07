import torch
import random
import copy
import fileutils
import model
import dataset

from math import sqrt

TEST_SAMPLE_LEN = 50


class NeuralNetTrainer:
    def __init__(self, config_file_path, experiment_dir_path):

        self.config = fileutils.get_config(config_file_path)
        print("Configuration:")
        print(self.config)
        
        self.experiment_dir_path = experiment_dir_path
        
        self.rand_seed = None     # Random seed for dividing training, validation and test dataset
        self.epoch = None         # Training iteration
        
        self.net = None           # Neural network
        self.best_net = None      # Best network
        
        self.criterion = None     # Objective function evaluator
        self.optimizer = None     # optimizer
        
        self.train_losses = None  # Training losses
        self.val_losses = None    # Validation losses
        self.best_score = None    # Best validation loss
        
        self.init_experiment(self.config)
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataset.get_dataloaders(self.config, self.rand_seed)
    
    # Initialize net, optimizer, criterion and stats
    def init_experiment(self, config):
        input_size = config["model"]["input_size"]
        hidden_size = config["model"]["hidden_size"]
        num_lstm_layers = config["model"]["num_lstm_layers"]
        len_decode_serie = config["model"]["len_decode_serie"]
        dropout_rate = config["model"]["dropout_rate"]

        learning_rate = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]

        self.net = model.LSTMNetV2(input_size, hidden_size, num_lstm_layers, len_decode_serie, dropout_rate)
        self.best_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=0)
        self.criterion = torch.nn.MSELoss(reduction="sum")
        
        self.train_losses = []
        self.val_losses = []
        self.best_score = float("inf")

        # Load saved state if there is one
        fileutils.load_experiment_state(self)
        
        print("Random seed: ", self.rand_seed)
        print("Current epoch:", self.epoch)

        # Use GPU for training
        if torch.cuda.is_available():
            self.net = self.net.cuda().float()
            self.best_net = self.best_net.cuda().float()
            self.criterion = self.criterion.cuda()
        else:
            raise("CUDA Not Available, CPU training not supported")

    # Training + validation loops
    def go(self):
        print("GO!")
        print("Best validation loss:", self.best_score)

        overfit_limit = self.config["training"]["overfit_limit"]
        num_epochs = self.config["training"]["num_epochs"]

        overfit = 0

        while self.epoch < num_epochs:
            print("Epoch ", self.epoch)
            
            train_loss = self.train()
            self.train_losses.append(train_loss)
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print("Training loss: ", train_loss, end='')
            print(" | Validation loss: ", val_loss)

            # Save best model and check overfitting
            if val_loss < self.best_score:
                self.best_net = copy.deepcopy(self.net)
                self.best_score = val_loss
                overfit = 0
            else:
                overfit += 1
                if overfit == overfit_limit:
                    print("Early stopping!")
                    break

            self.epoch += 1

        print("Best validation loss:", self.best_score)
            
        # Save model state and log statistics
        fileutils.save_experiment_state(self)
        fileutils.log_stats(self.train_losses, self.val_losses, self.experiment_dir_path)


    # Train an iteration through the training data in train_dataloader
    #   and optimize the neural net 
    # Return - averaged training loss
    def train(self):
        self.net.train()
        self.net.cuda()

        training_loss = 0
        
        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            self.optimizer.zero_grad()

            predictions = self.net(X, y)
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
        self.net.cuda()
        
        val_loss = 0
        
        with torch.no_grad():
            for i, (X, y) in enumerate(self.val_dataloader):
                X, y = X.cuda(), y.cuda()

                predictions = self.net(X, y)
                loss = self.criterion(predictions, y)

                val_loss += loss.item()

        # Average by data points and data serie length, take square root to get RMSD from MSE
        return sqrt(val_loss / len(self.val_dataloader.dataset) / y.shape[1])


    def test(self):
        self.best_net.eval()
#         torch.cuda.empty_cache()
#         self.best_net.cpu()

        test_loss = 0

        actual_serie = []
        predicted_serie = []

        with torch.no_grad():
            for i, (X, y) in enumerate(self.test_dataloader):
                X, y = X.cuda(), y.cuda()

                predictions = self.best_net(X)
                loss = self.criterion(predictions, y)

                for j in range(len(y)):
                    for k in range(len(y[0])):
                        actual_serie.append(y[j, k])
                        predicted_serie.append(predictions[j, k])

                test_loss += loss.item()
                
                if i == 0:
                    print("Sample batch data")
                    print("X shape: ", X.shape)
                    print("y shape: ", y.shape)

#         Print an example of prediction vs actual data
#         print("Example test data")
#         print("Predicted: \n", predictions)
#         print("Actual: \n", y)


        start = random.randint(0, len(actual_serie) - TEST_SAMPLE_LEN)
        zero = [0 for i in range(len(actual_serie))]
        fileutils.make_plot([actual_serie[start:start+TEST_SAMPLE_LEN], predicted_serie[start:start+TEST_SAMPLE_LEN]], 
                            ["Actual", "Predicted"], "Time step", 
                            "Rate of change", "test_data_regression",
                            self.experiment_dir_path)

        # Average by data points and data serie length, take square root to get RMSD from MSE
        test_loss = sqrt(test_loss / len(self.test_dataloader.dataset) / y.shape[1])
        print("Test loss:", test_loss)
