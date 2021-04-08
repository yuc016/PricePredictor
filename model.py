import torch
import torch.nn as nn

class LSTMNetV1(nn.Module):
    def __init__(self, input_feature_size, lstm_size, num_layers, output_feature_size, dropout_rate):
        super(LSTMNetV1, self).__init__()
        
        self.lstm = nn.LSTM(input_feature_size, lstm_size, num_layers, batch_first=True)

        self.lin2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(lstm_size, output_feature_size)
        )
        
        
    # input_series - (Batch size x serie length x feature size)
    # output_series - (Batch size x serie length x feature size)
    def forward(self, input_series, output_series, debug_print=False):
        decode_serie_len = output_series.shape[1]
        if decode_serie_len > 1:
            x = torch.cat([input_series, output_series], dim=1)
        else:
            x = input_series
#         print(x.shape)
        x, _ = self.lstm(x)
        # print(x.shape)
        x = x[:, -decode_serie_len:, :]
        # print(x.shape)
        output_series = self.lin2(x)
        # print(x.shape)
        # exit()
        return output_series


class LSTMNetV2(nn.Module):
    def __init__(self, input_feature_size, conv1_size, conv1_kernel_size, lstm_size, num_layers, l2_size, output_feature_size, dropout_rate):
        super(LSTMNetV2, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_size, out_channels=conv1_size, kernel_size=conv1_kernel_size, 
                        stride=1, padding=conv1_kernel_size//2, padding_mode='replicate'),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(conv1_size, lstm_size, num_layers, batch_first=True)

        self.lin2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(lstm_size, l2_size),
            nn.Dropout(dropout_rate * 0.5),
            nn.ReLU(),
            nn.Linear(l2_size, output_feature_size)
        )
        
    # input_series - (Batch size x serie length x feature size), [X, y]
    # output_series - (Batch size x serie length x feature size)
    def forward(self, input_series, debug_print=False):
        x = input_series
#         print(x.shape)
        x = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
#         print(x.shape)
        x, _ = self.lstm(x)
#         print(x.shape)
        x = x[:, -1:, :]
#         print(x.shape)
        output_series = self.lin2(x)
#         print(x.shape)
#         exit()
        return output_series
    

class LSTMNetV3(nn.Module):
    def __init__(self, input_feature_size, l1_size, conv1_size, lstm_size, num_layers, l2_size, output_feature_size, dropout_rate):
        super(LSTMNetV3, self).__init__()
        
        self.lin1 = nn.Sequential(
                        nn.Linear(input_feature_size, l1_size),
                        nn.Dropout(dropout_rate),
                        nn.ReLU()
                    )
            
        self.conv1d = nn.Sequential(
                    nn.Conv1d(in_channels=l1_size, out_channels=conv1_size, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
                    nn.Dropout(dropout_rate),
                    nn.ReLU()
                )
        
        self.lstm = nn.LSTM(conv1_size, lstm_size, num_layers, batch_first=True)

        self.lin2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, output_feature_size)
        )
        
    # input_series - (Batch size x serie length x feature size)
    # output_series - (Batch size x serie length x feature size)
    def forward(self, input_series, debug_print=False):
#         print(input_series.shape)
        x = self.lin1(input_series)
#         print(x.shape)
        x = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
#         print(x.shape)
        x, _ = self.lstm(x)
#         print(x.shape)
        x = x[:, -1:, :]
#         print(x.shape)
        output_series = self.lin2(x)
#         print(x.shape)
        return output_series
    


class LSTMNetV4(nn.Module):
    def __init__(self, input_feature_size, conv1_size, conv1_kernel_size, conv2_size, conv2_kernel_size, 
                    lstm_size, num_layers, l2_size, output_feature_size, dropout_rate):
        super(LSTMNetV4, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_size, out_channels=conv1_size, kernel_size=conv1_kernel_size, 
                        stride=1, padding=conv1_kernel_size//2, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=3, 
                        stride=1, padding=conv2_kernel_size//2, padding_mode='replicate'),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(conv2_size, lstm_size, num_layers, batch_first=True)

        self.lin2 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(lstm_size, l2_size),
            nn.Dropout(dropout_rate * 0.5),
            nn.ReLU(),
            nn.Linear(l2_size, output_feature_size)
        )
        
    # input_series - (Batch size x serie length x feature size), [X, y]
    # output_series - (Batch size x serie length x feature size)
    def forward(self, input_series, debug_print=False):
        x = input_series
#         print(x.shape)
        x = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
#         print(x.shape)
        x, _ = self.lstm(x)
#         print(x.shape)
        x = x[:, -1:, :]
#         print(x.shape)
        output_series = self.lin2(x)
#         print(x.shape)
#         exit()
        return output_series