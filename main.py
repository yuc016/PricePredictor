from trainer import NeuralNetTrainer
import os
import sys
import shutil

EXP_ROOT_DIR = "experiments"

if __name__ == "__main__":
    print()
    config_file_path = None
    experiment_dir_path = None
    experiment_name = "new_experiment"

    # Check if experiment name provided
    if len(sys.argv) == 2:
        experiment_name = sys.argv[1]

    print("Experiment name: ", experiment_name)

    # Create experiment folder if not exists
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)
    if not os.path.exists(experiment_dir_path):
        os.mkdir(experiment_dir_path)
        
    # Save current config to the experiment if config not exists
    config_file_path = os.path.join(experiment_dir_path, "config.json")
    if not os.path.isfile(config_file_path):
        print("config.json saved to the experiment directory")
        shutil.copy("config/config.json", config_file_path)

    trainer = NeuralNetTrainer(config_file_path, experiment_dir_path)
    trainer.load_data_for_training()
        
    trainer.go() # Training/validating loops
