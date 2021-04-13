import torch
import random
import copy
import fileutils
import dataset
import model_factory

from math import sqrt, exp


class NeuralNetTrainer:
    def __init__(self, config_file_path, experiment_dir_path, print_info=True):

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
        
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            
        self.print_info = print_info

        self.init_experiment(self.config)
        
    # Initialize net, optimizer, criterion and stats
    def init_experiment(self, config):
        
        learning_rate = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]
        
        self.net = model_factory.build_model(config)
        self.best_net = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=0)
        self.criterion = torch.nn.MSELoss(reduction="sum")
        
        self.train_losses = []
        self.val_losses = []
        self.best_score = float("inf")

        # Load saved state if there is one
        fileutils.load_experiment_state(self, self.cuda)

        if self.print_info:
            print("Random seed: ", self.rand_seed)
            print("Current epoch:", self.epoch)
            print("Best validation loss:", self.best_score)
            print()
            
        self.net = self.net.float()
        self.best_net = self.best_net.float()

        # Use GPU for training
        if self.cuda:
            self.net = self.net.cuda()
            self.best_net = self.best_net.cuda()
            self.criterion = self.criterion.cuda()


    def load_data_for_training(self):
        test_set_start_index = self.config["data"]["test_set_start_index"]
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
        return sqrt(training_loss / len(self.train_dataloader.dataset) / y.shape[1] / y.shape[2])


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
        return sqrt(val_loss / len(self.val_dataloader.dataset) / y.shape[1] / y.shape[2])


    # Run an iteration through the test data in test_dataloader
    #     and makes a graph of predicted vs actual time serie
    #     with assumption that the test data is in sequenced order
    #     e.g.   X = [[t1,t2,t3],      y = [t4,
    #                 [t2,t3,t4],           t5,
    #                 [t3,t4,t5],]          t6]
    def test(self, test_name=""):
        self.best_net.eval()
        if self.cuda:
            self.best_net.cuda()
        else:   
            self.best_net.cpu()

        test_loss = 0
        
        num_features = self.config["model"]["output_feature_size"]
        output_serie_len = self.config["data"]["output_serie_len"]

        actual_serie = torch.empty((len(self.test_dataloader.dataset) * output_serie_len, num_features))
        predicted_serie = torch.empty((len(self.test_dataloader.dataset) * output_serie_len, num_features))
        i = 0

        with torch.no_grad():
            for _, (X, y) in enumerate(self.test_dataloader):
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()
                
                predictions = self.best_net(X, y)
                loss = self.criterion(predictions, y)

                # Append test loss of each data point in order
#                 for j in range(len(y)):
#                     for k in range(len(y[0])):
#                         actual_serie.append(y[j, k].item())
#                         predicted_serie.append(predictions[j, k].item())
#                 for j in range(y.shape[0]):
#                     actual_serie.append(y[j, -1].item())
#                     predicted_serie.append(predictions[j, -1].item())
                for j in range(y.shape[0]):
                    actual_serie[i:i+output_serie_len] = y[j]
                    predicted_serie[i:i+output_serie_len] = predictions[j]
                    i += output_serie_len

                test_loss += loss.item()
            
#         print(predicted_serie)
                
        # Average by data points and data serie length, take square root to get RMSD from MSE
        test_loss = sqrt(test_loss / len(self.test_dataloader.dataset) / y.shape[1] / y.shape[2])
                
        # Make an actual vs prediction plot
        test_sample_len = self.config["testing"]["test_regression_sample_len"]
        for f in range(num_features):
            start = random.randint(0, len(actual_serie) - test_sample_len)
            zero = [0 for i in range(len(actual_serie))]
            fileutils.make_plot([actual_serie[start:start+test_sample_len,f], predicted_serie[start:start+test_sample_len,f]],
                                ["Actual", "Predicted"], "Time step", 
                                "Rate of change", "test_data_regression_feature"+str(f)+test_name,
                                self.experiment_dir_path)

        # Do a test trade
        product_net, my_net, trend_pred_accuracy = self.test_trade(self.config["testing"]["test_trade_start"],
                                                                    self.config["testing"]["test_trade_end"],
                                                                    actual_serie, predicted_serie, test_name)

        fileutils.write_experiment_result(self.experiment_dir_path, test_loss, product_net, my_net, trend_pred_accuracy)


    def single_predict(self, serie):
        self.best_net.eval()
        self.best_net.cpu().float()

        X = serie.unsqueeze(0)
        prediction = self.best_net(X).squeeze(0)
        
        return prediction[0].tolist()

                        
    # Perform test trade and save the result to experiment folder 
    def test_trade(self, trade_start, trade_end, actual_serie, predict_serie, test_name=""):
        if trade_start != -1 and trade_end != -1:
            actual_serie = actual_serie[trade_start:trade_end]
            predict_serie = predict_serie[trade_start:trade_end]
            
        # Start with 100 percent
        product_net_serie = [100.0]
        my_net_serie = [100.0]
        product_net = 100.0
        my_net = 100.0
        
        BUY_TRANSACTION_FEE_RATE = 0.0002
        SELL_TRANSACTION_FEE_RATE = 0.0002
        BUY_THRESH = 0
        bought_in = False
        correct_trend_pred_ct = 0
        
        for i in range(len(actual_serie)):
            # actual_roc = (-1) ** (actual_serie[i,-1] + 0.5) * -1 * exp(actual_serie[i, 0])
            actual_roc = actual_serie[i,-1] / 1000
            predict_roc = predict_serie[i, -1]

            if actual_roc * predict_roc > 0:
                correct_trend_pred_ct += 1

            # Update equity net worth
            product_net = product_net * (1 + actual_roc)

            # Buy in if predict growth
            if predict_roc > BUY_THRESH:
                if not bought_in:
                    my_net = my_net * (1 - BUY_TRANSACTION_FEE_RATE) # pay transaction fee
                    bought_in = True
                my_net = my_net * (1 + actual_roc) # value move with the product
            else:
                if bought_in:
                    my_net = my_net * (1 - SELL_TRANSACTION_FEE_RATE) # pay transaction fee
                    bought_in = False
            
            product_net_serie.append(product_net)
            my_net_serie.append(my_net)
            
#         for i in range(len(actual_serie)):
#             pred_high = 1 + (predict_serie[i,0].item() / 1000)
#             pred_low = 1 + (predict_serie[i,1].item() / 1000)
#             pred_close = 1 + (predict_serie[i,2].item() / 1000)
#             actual_high = 1 + (actual_serie[i,0].item() / 1000)
#             actual_low = 1 + (actual_serie[i,1].item() / 1000)
#             actual_close = 1 + (actual_serie[i,2].item() / 1000)
            
# #             pred_high *= 0.99999
# #             pred_low *= 1.00001
            
#             # Update equity net worth
#             product_net = product_net * actual_close
            
#             if not bought_in:
#                 # Limit buy at predicted low price, execute if actual low price falls below
#                 if actual_low < pred_low:
#                     my_net *= (actual_close / pred_low)
#                     bought_in = True
#             else:
#                 # Limit sell at predicted high price, execute if actual high price rises above
#                 if actual_high > pred_high:
#                     my_net *= pred_high
#                     bought_in = False
#                 else:
#                     my_net *= actual_close

#             product_net_serie.append(product_net)
#             my_net_serie.append(my_net)

        fileutils.make_plot([product_net_serie, my_net_serie],
                            ["Actual Product Worth", "My Net Worth"], 
                            "Time step", "Percentage of Original Capital", 
                            "test_trade"+test_name, self.experiment_dir_path)
        
        return product_net, my_net, correct_trend_pred_ct / len(actual_serie)
