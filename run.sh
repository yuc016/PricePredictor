#!/bin/bash

source "/home/yuc016/PricePredictor/chrome_headless/.venv/bin/activate"
python fetch_data.py
deactivate
python predict.py $1
