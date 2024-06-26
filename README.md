What is this project?
 
- This project is developed to predict Bitcoin price with a CNN-LSTM deep neural net, but it is easily configurable and extensible to perform any time serie forecasting task. It features a structured pipeline for data pre-processing, model training/saving, performance logging/reporting and live testing/predicting.
  
What can I do with this project?

- You can find your own data and plug them into the network to see how well it performs. You can tune the parameters of the existing models or experiment with new models to make predictions better. You can hour-trade Bitcoin using the existing framework.

Show me steps to get your thing running!

- First, download the data I'm using: https://www.kaggle.com/mczielinski/bitcoin-historical-data to data/kaggle/ folder. This data is made by Zielak on Kaggle.com.
- Set configuration in config/config.json. If in doubt, check model.py to better understand the meaning of the parameters.
- Run the make_training_data python notebook in data/kaggle/ folder to get your X.pt and y.pt training data
