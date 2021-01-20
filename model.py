import torch
import torch.nn as nn


class PPNetV1(nn.Module):
    def __init__(self, hidden_size):
        super(PPNetV1, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size)
        
    # input_series - (Batch size x sequence length x 1)
    # target_series - (Batch size x sequence length x 1)
    def forward(self, input_series, target_series):
        time_series = torch.cat((input_series, target_series), dim=1)
        predict_series, _, _ = self.lstm(time_series)
        
        target_serie_len = target_series.shape[1]

        # Return predictions at target serie time steps
        return predict_series[-target_serie_len-1:-1]