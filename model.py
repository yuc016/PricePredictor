import torch
import torch.nn as nn


class PPNetV1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PPNetV1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.Linear(hidden_size, output_size)
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )
        
    # input_series - (Batch size x sequence length x input_size)
    # target_series - (Batch size x sequence length x input_size)
    def forward(self, input_series, target_series, debug_print=False):
        # Pass the entire serie through lstm
        input_time_series = torch.cat((input_series, target_series), dim=1)
        features, _ = self.lstm(input_time_series)

        # Truncate the series to last few timesteps with length of target series
        output_serie_len = target_series.shape[1]
        features = features[:,-output_serie_len-1:-1,:]

        # Linear layer to reduce feature vectors
        predicted_series = self.linear(features)

        if debug_print:
            print("input_series shape:", input_series.shape)
            print("target_series shape:", target_series.shape)
            print("input_time_series shape:", input_time_series.shape)
            print("features shape:", features.shape)
            print("predicted_series shape:", predicted_series.shape)

        # Return predictions at target serie time steps
        return predicted_series