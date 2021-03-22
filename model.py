import torch
import torch.nn as nn


class LSTMNetV1(nn.Module):
    def __init__(self, input_size, l1_size, conv1_size, lstm_size, num_layers, l2_size, output_size, dropout_rate):
        super(LSTMNetV1, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=conv1_size, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_size, out_channels=conv1_size * 2, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers, batch_first=True)

        self.lin1 = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Flatten(),
#             nn.Linear(lstm_size, l2_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_size, output_size)
        )
        
        
    # input_series - (Batch size x sequence length x input_size)
    def forward(self, input_series, debug_print=False):
        x = input_series
#         print(x.shape)
#         x = self.conv1d(input_series.permute(0,2,1)).permute(0,2,1)
#         print(x.shape)
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x[:, -1:, :].squeeze(1)
        # print(x.shape)
        x = self.lin1(x)
        # print(x.shape)
        # exit()
        return x
    

class LSTMNetV2(nn.Module):
    def __init__(self, input_size, l1_size, conv1_size, lstm_size, num_layers, l2_size, output_size, dropout_rate):
        super(LSTMNetV2, self).__init__()
        
        self.lin1 = nn.Sequential(
                        nn.Linear(input_size, l1_size),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    )
            
        self.conv1d = nn.Sequential(
                    nn.Conv1d(in_channels=l1_size, out_channels=conv1_size, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
        
        self.lstm = nn.LSTM(conv1_size, lstm_size, num_layers, batch_first=True)

        self.lin2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, output_size)
        )
        
    # input_series - (Batch size x sequence length x input_size)
    def forward(self, input_series, debug_print=False):
#         print(input_series.shape)
        x = self.lin1(input_series)
#         print(x.shape)
        x = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
#         print(x.shape)
        x, _ = self.lstm(x)
#         print(x.shape)
        x = x[:, -1:, :].squeeze(1)
#         print(x.shape)
        x = self.lin2(x)
#         print(x.shape)
        return x
    