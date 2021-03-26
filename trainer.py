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

        self.config = fileutils.get_config_from_file(config_file_path)
        
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

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.init_experiment(self.config)
        
    # Initialize net, optimizer, criterion and stats
    def init_experiment(self, config):
        
        model_name = config["model"]["name"]
        input_size = config["model"]["input_size"]
        l1_size = config["model"]["l1_size"]
        conv1_size = config["model"]["conv1_size"]
        l2_size = config["model"]["l2_size"]
        lstm_size = config["model"]["lstm_size"]
        num_lstm_layers = config["model"]["num_lstm_layers"]
        len_decode_serie = config["model"]["len_decode_serie"]
        dropout_rate = config["model"]["dropout_rate"]

        learning_rate = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]
        
        if model_name == "LSTMNetV1":
            self.net = model.LSTMNetV1(input_size, lstm_size, num_lstm_layers, len_decode_serie, dropout_rate)
        elif model_name == "LSTMNetV2":
            self.net = model.LSTMNetV2(input_size, conv1_size, lstm_size, num_lstm_layers, l2_size, len_decode_serie, dropout_rate)
        elif model_name == "LSTMNetV3":
            self.net = model.LSTMNetV3(input_size, l1_size, conv1_size, lstm_size, num_lstm_layers, l2_size, len_decode_serie, dropout_rate)
        else:
            raise("Unknown model name", model_name)
            
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
        print("Best validation loss:", self.best_score)
        print()

        # Use GPU for training
        if torch.cuda.is_available():
            self.net = self.net.cuda().float()
            self.best_net = self.best_net.cuda().float()
            self.criterion = self.criterion.cuda()
        else:
            raise("CUDA Not Available, CPU training not supported")
    

    def load_data_for_training(self):
        test_set_start_index = self.config["dataset"]["test_set_start_index"]
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataset.get_dataloaders(self.config, self.rand_seed, test_set_start_index)
    

    # AI learning main loop
    def go(self):
        # Do a pre-test for a new experiment
        if self.best_score == float("inf"):
            self.test("_init")

        num_epochs = self.config["training"]["num_epochs"]

        overfit_limit = self.config["training"]["overfit_limit"]
        overfit_ct = 0
        
        save_ct = 0
        save_period = self.config["experiment"]["save_period"]

        print("\nGO!\n")

        # Training and validation loop
        while self.epoch < num_epochs:
            print("Epoch ", self.epoch)
            
            train_loss = self.train()
            self.train_losses.append(train_loss)
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print("Training loss: ", '{0:.5}'.format(train_loss), end='')
            print(" ---- Validation loss: ", '{0:.5}'.format(val_loss))

            # Save best model and check overfitting
            if val_loss < self.best_score:
                self.best_net = copy.deepcopy(self.net)
                self.best_score = val_loss
                overfit_ct = 0
            else:
                overfit_ct += 1
                if overfit_ct == overfit_limit:
                    print("Early stopping!")
                    break

            self.epoch += 1
            
            # Save experiment state and log statistics periodically
            save_ct += 1
            if save_ct % save_period == 0:
                save_ct = 0
                fileutils.save_experiment_state(self)
                fileutils.log_loss_stats(self.train_losses, self.val_losses, self.experiment_dir_path)

        print("Best validation loss:", self.best_score)
        
        # Post training test
        self.test()

        # Save experiment state
        fileutils.save_experiment_state(self)
        fileutils.log_loss_stats(self.train_losses, self.val_losses, self.experiment_dir_path)

            

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


    # Run an iteration through the test data in test_dataloader
    #     and makes a graph of predicted vs actual time serie
    #     with assumption that the test data is in sequenced order
    #     e.g.   X = [[t1,t2,t3],      y = [t4,
    #                 [t2,t3,t4],           t5,
    #                 [t3,t4,t5],]          t6]
    def test(self, test_name=""):
        self.best_net.eval()
        self.best_net.cpu()

        test_loss = 0

        actual_serie = []
        predicted_serie = []

        with torch.no_grad():
            for i, (X, y) in enumerate(self.test_dataloader):

                predictions = self.best_net(X)
                loss = self.criterion(predictions, y)

                # Append test loss of each data point in order
                for j in range(len(y)):
                    for k in range(len(y[0])):
                        actual_serie.append(y[j, k].item())
                        predicted_serie.append(predictions[j, k].item())

                test_loss += loss.item()
                
                # # Print an example of prediction vs actual data
                # if i == 0:
                #     print("Sample batch data")
                #     print("X shape: ", X.shape)
                #     print("y shape: ", y.shape)
                #     print(type(X[0,0].item()))

        # Average by data points and data serie length, take square root to get RMSD from MSE
        test_loss = sqrt(test_loss / len(self.test_dataloader.dataset) / y.shape[1])
        print("Test loss:", test_loss, "\n")
                
        # Make a actual vs prediction plot
        start = random.randint(0, len(actual_serie) - TEST_SAMPLE_LEN)
        zero = [0 for i in range(len(actual_serie))]
        fileutils.make_plot([actual_serie[start:start+TEST_SAMPLE_LEN], predicted_serie[start:start+TEST_SAMPLE_LEN]],
                            ["Actual", "Predicted"], "Time step", 
                            "Rate of change", "test_data_regression"+test_name,
                            self.experiment_dir_path)

        # Do a test trade
        self.test_trade(actual_serie, predicted_serie, test_name)


    def single_predict(self, serie):
        self.best_net.eval()
        self.best_net.cpu().float()

        X = serie.unsqueeze(0)
        prediction = self.best_net(X).squeeze(0)

        return prediction.item()

                        
    # Perform test trade and save the result to experiment folder 
    def test_trade(self, actual_serie, predicted_serie, test_name=""):
        # Start with 100 percent
        product_net_serie = [100]
        my_net_serie = [100]
        product_net = 100
        my_net = 100
        
        bought_in = False
        
        for i in range(len(actual_serie)):
            # If predict growth, buy in
            bought_in = True if predicted_serie[i] > 0 else False
            
            # Update equity net worth
            product_net = product_net * (1 + actual_serie[i] / 1000)
            
            # my net worth move with actual change if bought in
            if bought_in:
                my_net = my_net * (1 + actual_serie[i] / 1000)
            
            product_net_serie.append(product_net)
            my_net_serie.append(my_net)
            
        product_net_change = product_net - 100
        my_net_change = my_net - 100

        print("Product net change (throughout the period): {:.2f}%".format(product_net_change))
        print("My net change (trading by predicting at every time step): {:.2f}%".format(my_net_change))
        print()
        
        fileutils.make_plot([product_net_serie, my_net_serie],
                            ["Actual Product Worth", "My Net Worth"], 
                            "Time step", "Percentage of Original Capital", 
                            "test_trade"+test_name, self.experiment_dir_path)
