import os
import sys
import signal
import time
import robin_stocks.robinhood as r

from run_live import run_live

EXP_ROOT_DIR = "experiments"
SYMBOL = 'BTC'
TIME_INTERVAL = 60
BUY_THRESH = 0
ACCOUNT = 'chenyyo0o0o@gmail.com'
CANCEL_BUFFER = 5
TRANSACTION_FEE_RATE = 0.0001

def sigint_exit(sig, frame):
    print('Logging out..')
    r.cancel_all_crypto_orders()
    r.logout()    
    print('Cancelled all pending orders and logged out')
    sys.exit(0)
    

def trade_loop(config_file_path, experiment_dir_path):
    while True:
        now = int(time.time())
        # wait until next time step
        while (now + CANCEL_BUFFER) % (TIME_INTERVAL // 3) != 0:
            time.sleep(0.5)
            now = int(time.time())
            
        print("\n\n\n")

        new_timestep = (now + CANCEL_BUFFER) % TIME_INTERVAL == 0
        if new_timestep:
            print("New timestep")
            
        # Cancel pending order
        print("Cancelling pending orders")
        r.cancel_all_crypto_orders()

        # Wait until server updates new data
        time.sleep(CANCEL_BUFFER)

        last_close, predicted_close, predicted_roc = run_live(config_file_path, experiment_dir_path, mode='p', logged_in=True)
        print("Predicted ROC:", predicted_roc)
        
        if predicted_roc > BUY_THRESH:
            capital = float(r.load_phoenix_account('crypto_buying_power')['amount'])
            
            mark_price = float(r.get_crypto_quote(SYMBOL, 'mark_price'))
            print("Mark price: ", mark_price)
            
            if new_timestep:
                limit_price = min(last_close, mark_price)
                limit_price = mark_price
            else:
                limit_price = mark_price
            limit_price *= (1 + TRANSACTION_FEE_RATE)
            print("Buy limit price:", limit_price)
            
            # r.order_buy_crypto_limit_by_price(SYMBOL, capital, limit_price)
        else:
            quantity_held = r.get_crypto_quantity_held(SYMBOL)
            
            mark_price = float(r.get_crypto_quote(SYMBOL, 'mark_price'))
            print("Mark price: ", mark_price)
            
            if new_timestep:
                limit_price = max(last_close, mark_price)
                limit_price = mark_price
            else:
                limit_price = mark_price
            limit_price *= (1 - TRANSACTION_FEE_RATE)
            print("Sell limit price:", limit_price)
            
            # r.order_sell_crypto_limit(SYMBOL, quantity_held, limit_price)

        print("Action complete!")

        
if __name__ == "__main__":
    print()
    config_file_path = None
    experiment_dir_path = None

    # Must provide an experiment directory
    if len(sys.argv) < 2:
        raise Exception("Usage: python robin_trade.py <experiment_name>\n")

    # Check experiment exists
    experiment_name = sys.argv[1]
    experiment_dir_path = os.path.join(EXP_ROOT_DIR, experiment_name)
    if not os.path.exists(experiment_dir_path):
        raise Exception(experiment_dir_path, " doesn't exist:")
    
    # Check config file exists
    config_file_path = os.path.join(experiment_dir_path, "config.json")
    if not os.path.isfile(config_file_path):
        raise Exception("config.json doesn't exist:")
        
    r.login(username=ACCOUNT, expiresIn=172800)
    print('Logged in')
    
    # Register signal interrupt behavior: log out
    signal.signal(signal.SIGINT, sigint_exit)

    trade_loop(config_file_path, experiment_dir_path)
    