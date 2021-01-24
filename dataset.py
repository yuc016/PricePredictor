import torch
import random
from torch.utils.data import DataLoader, Dataset

# Get saved X and y tensor from path
def get_data_tensor_from_path(X_file_path, y_file_path):
    return torch.load(X_file_path), torch.load(y_file_path)


# Split the data tensor X and corresponding tensor y into two sets
#   set 1 starts at random index of X and is continuous
# Return - (X1, X2), (y1, y2)
def split_data(X, y, set_1_percentage):
    set_1_size = int(X.shape[0] * set_1_percentage)
    set_2_size = X.shape[0] - set_1_size

    set_1_start = random.randint(0, len(X) - set_1_size)
    set_1_end = set_1_start + set_1_size

    set_1_X = X[set_1_start:set_1_end]
    set_1_y = y[set_1_start:set_1_end]
    set_2_X = torch.cat((X[:set_1_start], X[set_1_end:]), dim=0)
    set_2_y = torch.cat((y[:set_1_start], y[set_1_end:]), dim=0)

    return (set_1_X, set_2_X), (set_1_y, set_2_y)


# Shuffle data tensor X and corresponding tensor y in parallel
# Return - X, y
def shuffle_data(X, y):
    X = X.tolist()
    y = y.tolist()

    xy = list(zip(X,y))
    random.shuffle(xy)
    X, y = list(zip(*xy))

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    return X, y


def min_max_normalize(X, y):
    # Min max normalize the change
    min_value = torch.min(X)
    min_value = min(torch.min(y), min_value)
    max_value = torch.max(X)
    max_value = max(torch.max(y), max_value)

    X = torch.log2((X - min_value) / (max_value - min_value) + 1e-4)
    y = torch.log2((y - min_value) / (max_value - min_value) + 1e-4)

    return X, y


def get_dataloaders(config, rand_seed):
    X_file_path = config["dataset"]["X_file_path"]
    y_file_path = config["dataset"]["y_file_path"]
    test_set_percentage = config["dataset"]["test_set_percentage"]
    val_set_percentage = config["dataset"]["val_set_percentage"]
    batch_size = config["training"]["batch_size"]

    random.seed(rand_seed)

    X, y = get_data_tensor_from_path(X_file_path, y_file_path)
    (X_test, X_train), (y_test, y_train) = split_data(X, y, test_set_percentage)
    X_train, y_train = shuffle_data(X_train, y_train)
    (X_val, X_train), (y_val, y_train) = split_data(X_train, y_train, val_set_percentage)

    train_dataset = price_series_dataset(X_train, y_train)
    val_dataset = price_series_dataset(X_val, y_val)
    test_dataset = price_series_dataset(X_test, y_test)

    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class price_series_dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]