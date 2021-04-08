import robin_stocks.robinhood as r

from datetime import datetime

import pandas as pd
import torch


def make_input(time_interval, input_size, output_size, encode_length, decode_length, logged_in=False):
    if not logged_in:
        r.login(username='chenyyo0o0o@gmail.com')
    
    json_data = r.get_crypto_historicals('BTC', '10minute', 'week')
    
    if not logged_in:
        r.logout()
    
    list_data = []
    for i in range(len(json_data)):
        json_entry = json_data[i]
        datapoint = [json_entry['open_price'], json_entry['high_price'], json_entry['low_price'], json_entry['close_price']]
        list_data.append(datapoint)

    time_series_df = pd.DataFrame(list_data).astype('float')
    time_series_tensor = torch.tensor(time_series_df.values)

    columns = ["open", "high", "low", "close"]
    col_i = {}
    for i in range(len(columns)):
        col_i[columns[i]] = i 

    last_timestamp = json_data[-1]['begins_at']
    last_timestamp = last_timestamp.replace('Z', '')
    ymd, hms = last_timestamp.split('T')
    y, mo, d = ymd.split('-')
    h, mi, s = hms.split(':')
    last_timestamp = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s)).timestamp()
    last_close = time_series_tensor[-1, col_i["close"]]

    closing_price_roc = (time_series_tensor[1:, col_i["close"]] - time_series_tensor[:-1, col_i["close"]]) / time_series_tensor[:-1, col_i["close"]] * 1000
    closing_price_roc = closing_price_roc.reshape(-1,1)

    candle_stats = (time_series_tensor[1:, :] / time_series_tensor[:-1, col_i["close"]].reshape(-1,1) - 1) * 1000
    
    # Check for nans
    assert(torch.sum(closing_price_roc != closing_price_roc) == 0)
    assert(torch.sum(candle_stats != candle_stats) == 0)

    data_in = candle_stats
    data_out = closing_price_roc

    assert(data_in.shape[1] == input_size)
    assert(data_out.shape[1] == output_size)

    # Make input time serie data and target time serie data tensor 
    num_data = (data_in.shape[0] - encode_length - decode_length) // decode_length + 1

    X_shape = (num_data, encode_length, input_size)
    y_shape = (num_data, decode_length, output_size)
    pred_serie_shape = (encode_length, input_size)

    X, y = torch.empty(X_shape), torch.empty(y_shape)
    pred_serie = torch.empty(pred_serie_shape)

    for i in range(num_data):
        start = i * decode_length
        X[i] = data_in[start:start+encode_length]
        y[i] = data_out[start+encode_length:start+encode_length+decode_length]

    pred_serie[:, :] = data_in[-encode_length:]

    return X, y, pred_serie, last_timestamp, last_close
