from trainer import PPNeuralTrainer
import sys

if __name__ == "__main__":
    config_name = "config1"

    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    print("Using Configuration: ", config_name + ".json")
    trainer = PPNeuralTrainer(config_name)
    trainer.go()