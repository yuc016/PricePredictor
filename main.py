from trainer import NeuralNetTrainer
import os
import sys

EXP_ROOT_DIR = "./experiment"
CONFIG_ROOT_DIR = "./config"

if __name__ == "__main__":
    config_file_path = None
    experiment_dir_path = None
    experiment_name = "new_experiment"

    # Must provide a config file
    if len(sys.argv) == 1:
        raise Exception("Usage: python main.py <config> <experiment_name>")

    # Check config file exists
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config_file_path = os.path.join(CONFIG_ROOT_DIR, config_name + ".json")
        if not os.path.isfile(config_file_path):
            raise Exception(config_file_path, " doesn't exist:")
    
    # Check if experiment name provided
    if len(sys.argv) > 2:
        experiment_name = sys.argv[2]
    
    # Create experiment folder if not exists
    experiment_exists = True
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)
    if not os.path.exists(experiment_dir_path):
        experiment_exists = False
        os.mkdir(experiment_dir_path)
        

    print("Using configuration: ", config_name)
    print("Experiment name: ", experiment_name)

    trainer = NeuralNetTrainer(config_file_path, experiment_dir_path)
    if not experiment_exists:
        trainer.test("_init")
        # Save a copy of config file after test run to validate config settings
        os.system("cp " + config_file_path + " " + os.path.join(experiment_dir_path, config_name))
        
    trainer.go() # Training/validating loops
    trainer.test()