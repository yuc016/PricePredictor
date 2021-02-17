import torch
import torch.nn as nn


class PPNetV1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(PPNetV1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
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

class PPNetV2(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
#         super(PPNetV2, self).__init__()
#         self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

#         self.decoder = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
        
#     # input_series - (Batch size x sequence length x input_size)
#     def forward(self, input_series, debug_print=False):
#         encoder_out, _ = self.encoder(input_series)
#         # Output from last timestep
#         encoder_out = encoder_out[:, -1:, :].squeeze(1)
#         decoder_out = self.decoder(encoder_out)
        
#         return decoder_out

    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(PPNetV2, self).__init__()
        self.enhance_serie = nn.Sequential(
                    nn.Conv1d(in_channels=2, out_channels=16, kernel_size=7, stride=1, padding=3, padding_mode='replicate'),
#                     nn.ReLU(),
#                     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
                    nn.ReLU()
                )
        
        self.encoder = nn.LSTM(16, hidden_size, num_layers, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
#         self.encoder1 = nn.LSTM(64, 64, 1, batch_first=True)
#         self.shrink1 = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.ReLU()
#         )
#         self.encoder2 = nn.LSTM(64, 64, 1, batch_first=True)
#         self.shrink2 = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.ReLU()
#         )
#         self.encoder3 = nn.LSTM(64, 32, 1, batch_first=True)
#         self.shrink3 = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.ReLU(),
#             nn.Linear(32, output_size)
#         )

        
    # input_series - (Batch size x sequence length x input_size)
    def forward(self, input_series, debug_print=False):
        x = self.enhance_serie(input_series.permute(0,2,1)).permute(0,2,1)
        x, _ = self.encoder(x)
        x = x[:, -1:, :].squeeze(1)
        x = self.decoder(x)
        return x
        
#         encoder_out, _ = self.encoder1(encoder_out)
#         encoder_out = self.shrink1(encoder_out)
#         encoder_out, _ = self.encoder2(encoder_out)
#         encoder_out = self.shrink2(encoder_out)
#         encoder_out, _ = self.encoder3(encoder_out)
#         encoder_out = self.shrink3(encoder_out)
        
#         encoder_out = encoder_out[:, -1:, :].squeeze(1)
#         return encoder_out
