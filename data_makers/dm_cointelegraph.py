import os
import shutil
import time
from datetime import datetime

import csv
import pandas as pd
import torch

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

DATA_ROOT_DIR = "data/live" 
FILE_NAME = "amCharts.csv"
TIME_INTERVAL = 3600

def make_data(encode_length, decode_length, data_file_path):
    fetch_data(data_file_path)
    return get_xy_tensors(encode_length, decode_length, data_file_path)


def fetch_data(data_file_path):
    options = webdriver.ChromeOptions()

    options.headless = True
    options.add_argument("--no-sandbox")
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.37"')
    
    driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=options)

    driver.execute_cdp_cmd('Page.setDownloadBehavior', {
        'behavior': 'allow',
        'downloadPath': os.getcwd()
    })

    # Reach website
    driver.get('https://cointelegraph.com/bitcoin-price-index')
    print("Loading website...")
    time.sleep(5)

    # print("Title: %s" % driver.title)
    # html = driver.page_source
    # html = ''.join((filter(lambda x: ord(x) >= 0 and ord(x) < 128, html)))
    # print(html)
    
    # Remove a element blocking the menu
    driver.execute_script("document.getElementsByClassName('privacy-policy')[0].remove();")

    # Click the menu
    export_menu = driver.find_elements_by_class_name('export-main')[1]
    export_menu.click()

    # Click save as
    save_btn = export_menu.find_element_by_xpath('ul/li[2]')
    save_btn.click()

    # Click csv
    csv_btn = save_btn.find_element_by_xpath('ul/li[1]')
    csv_btn.click()

    print("Waiting for file to be downloaded...")
    time.sleep(2)
    driver.close()

    shutil.move(FILE_NAME, data_file_path)


def get_xy_tensors(encode_length, decode_length, data_file_path):
    column = "close"

    # Read csv file and parse data into tensor
    time_series = None
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        time_series = list(reader)
    
    time_series_df = pd.DataFrame(time_series[1:], columns=time_series[0])

    # Assert that no data is missing
    time_stamps = time_series_df["date"]
    time_stamps_int = [0 for i in range(len(time_stamps))]
    # Convert string time to int
    for i in range(len(time_stamps)):
        time_str = time_stamps[i]
        ymd, t = time_str.split(' ')
        y, mo, d = ymd.split('-')
        h, mi, s = t.split(':')
        
        # Stupid data, stupid fix 24:00:00 wtf?????????
        if int(h) == 24:
            h = str(0)
            d = str(int(d) + 1)
        time_stamps_int[i] = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s)).timestamp()
    for i in range(len(time_stamps_int) - 1):
        print(time_stamps_int[i])
        print(time_stamps_int[i+1])
        assert(time_stamps_int[i] + TIME_INTERVAL == time_stamps_int[i+1])
    # UTC - 8hr = PST
    print("Time of latest data point: ", time_stamps[i+1])
    print()

    time_series_df = time_series_df[[column]]
    time_series_df = time_series_df.astype(float)

    time_series_tensor = torch.tensor(time_series_df[column].values)

    mini = torch.min(time_series_tensor).item()
    maxi = torch.max(time_series_tensor).item()
    mean = torch.mean(time_series_tensor).item()

    print("Data summary: ")
    print("\tMax ", maxi)
    print("\tMin ", mini)
    print("\tMean ", mean)
    print()

    # Convert series data to change in percentage in one thousanth
    series_rate_change = (time_series_tensor[1:] - time_series_tensor[:-1]) / time_series_tensor[:-1]
    series_rate_change *= 1000

    mini = torch.min(series_rate_change).item()
    maxi = torch.max(series_rate_change).item()
    mean = torch.mean(series_rate_change).item()

    print("Rate of change (in 1000th):")
    print("\tMax", maxi)
    print("\tMin", mini)
    print("\tMean", mean)
    print()

    # Check for nans
    assert(torch.sum(series_rate_change != series_rate_change) == 0)

    data = series_rate_change.reshape(-1, 1)

    # Make input time serie data and target time serie data tensor 
    num_data = (len(data) - encode_length - decode_length) // decode_length + 1

    X_shape = (num_data, encode_length, len(data[0]))
    y_shape = (num_data, decode_length)

    X, y = torch.empty(X_shape), torch.empty(y_shape)

    for i in range(num_data):
        start = i * decode_length
        X[i] = data[start:start+encode_length]
        y[i] = data[start+encode_length:start+encode_length+decode_length, -1] # Price is last feature

    print("Number of data: ", str(len(X)))
    print()

    return X, y