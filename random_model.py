import random
import os
import fileutils

CONFIG_FILE_PATH = "config/config.json"

for i in range(60, 99):
    config = fileutils.get_config_from_file(CONFIG_FILE_PATH)
    config["model"]["conv1_size"] = random.randint(150, 450)
    config["model"]["conv1_kernel_size"] = random.randint(2, 8) * 2 - 1
    config["model"]["lstm_size"] = random.randint(200, 800)
    config["model"]["num_lstm_layers"] = random.randint(1, 4)
    config["model"]["l2_size"] = random.randint(64, 256)
    fileutils.write_config_to_file(CONFIG_FILE_PATH, config)

    os.system("python main.py " + str(i))

    # Set num epochs to avoid further training
    experiment_config_file_path = os.path.join("experiments/" + str(i), "config.json")
    config = fileutils.get_config_from_file(experiment_config_file_path)
    config["training"]["num_epochs"] = 200
    fileutils.write_config_to_file(experiment_config_file_path, config)


