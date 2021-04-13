import model


def build_model(config):
    model_name = config["model"]["name"]
    input_serie_len = config["data"]["input_serie_len"]
    output_serie_len = config["data"]["output_serie_len"]
    input_feature_size = config["model"]["input_feature_size"]
    output_feature_size = config["model"]["output_feature_size"]
    l1_size = config["model"]["l1_size"]
    conv1_size = config["model"]["conv1_size"]
    conv1_kernel_size = config["model"]["conv1_kernel_size"]
    conv2_size = config["model"]["conv2_size"]
    conv2_kernel_size = config["model"]["conv2_kernel_size"]
    l2_size = config["model"]["l2_size"]
    lstm_size = config["model"]["lstm_size"]
    num_lstm_layers = config["model"]["num_lstm_layers"]
    dropout_rate = config["model"]["dropout_rate"]

    if model_name == "PLAIN_LSTM":
        return model.PLAIN_LSTM(input_feature_size, lstm_size, num_lstm_layers, output_feature_size, dropout_rate)
    elif model_name == "CNN1L_LSTM":
        return model.CNN1L_LSTM(input_feature_size, conv1_size, conv1_kernel_size, 
                                    lstm_size, num_lstm_layers, l2_size, output_feature_size, dropout_rate)
    elif model_name == "LIN_CNN1L_LSTM":
        return model.LIN_CNN1L_LSTM(input_feature_size, l1_size, conv1_size, lstm_size, 
                                    num_lstm_layers, l2_size, output_feature_size, dropout_rate)
    elif model_name == "CNN2L_LSTM":
        return model.CNN2L_LSTM(input_feature_size, conv1_size, conv1_kernel_size, 
                                    conv2_size, conv2_kernel_size, lstm_size, num_lstm_layers, 
                                    l2_size, output_feature_size, dropout_rate)
    elif model_name == "LSTM_CNN":
        return model.LSTM_CNN(input_feature_size, conv1_size, conv1_kernel_size, 
                                    conv2_size, conv2_kernel_size, lstm_size, num_lstm_layers, 
                                    l2_size, output_feature_size, dropout_rate)
    else:
        raise("Unknown model name", model_name)