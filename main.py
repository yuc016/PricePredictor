from trainer import PPNeuralTrainer
import os
import sys

EXP_ROOT_DIR = "./experiment"
CONFIG_ROOT_DIR = "./config"

if __name__ == "__main__":
    config_file_path = None
    experiment_dir_path = None
    experiment_name = "new_experiment$$"

    if len(sys.argv) == 1:
        raise Exception("Please provide a <config.json> file to use in config directory")

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        config_file_path = os.path.join(CONFIG_ROOT_DIR, config_name + ".json")

        if not os.path.isfile(config_file_path):
            raise Exception("File doesn't exist ", config_file_path)
    
    if len(sys.argv) > 2:
        experiment_name = sys.argv[2]
        
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)
    if not os.path.exists(experiment_dir_path):
        os.mkdir(experiment_dir_path)

    print("Using Configuration: ", config_name + ".json")
    print("Experiment name: ", experiment_name)
    trainer = PPNeuralTrainer(config_file_path, experiment_dir_path)
    trainer.go()
    test_loss = trainer.test()
    print("Final test loss:", test_loss)