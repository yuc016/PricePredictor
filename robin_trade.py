import os
import sys
import signal
import time
import robin_stocks.robinhood as r

from run_live import run_live

EXP_ROOT_DIR = "experiments"
SYMBOL = 'BTC'
TIME_INTERVAL = 600
BUY_THRESH = 0
ACCOUNT = 'chenyyo0o0o@gmail.com'

def sigint_exit(sig, frame):
    print('Logging out..')
    r.cancel_all_crypto_orders()
    r.logout()    
    print('Cancelled all pending orders and logged out')
    sys.exit(0)
    

def trade_loop(config_file_path, experiment_dir_path):
    while True:
        # wait until next time step
        while int(time.time()) % TIME_INTERVAL != 0:
            # Cancel pending order 10 or 9 seconds before next time step
            if (int(time.time()) + 10) % TIME_INTERVAL == 0 or (int(time.time()) + 9) % TIME_INTERVAL == 0:
                print("\n\n\nCancelling pending orders")
                r.cancel_all_crypto_orders()
                time.sleep(3)
            time.sleep(0.8)
#             print("wait")

        # Wait 2 seconds to let server update
        time.sleep(2)

        last_close, predicted_close, predicted_roc = run_live(config_file_path, experiment_dir_path, mode='p', logged_in=True)
        print("Predicted ROC:", predicted_roc)
        
        if predicted_roc > BUY_THRESH:
            capital = float(r.load_phoenix_account('crypto_buying_power')['amount'])
            mark_price = float(r.get_crypto_quote(SYMBOL, 'mark_price'))
            print("Mark price: ", mark_price)
            limit_price = min(last_close, mark_price)
            print("Buy limit price:", limit_price)
            r.order_buy_crypto_limit_by_price(SYMBOL, capital, limit_price)
        else:
            quantity_held = r.get_crypto_quantity_held(SYMBOL)
            mark_price = float(r.get_crypto_quote(SYMBOL, 'mark_price'))
            print("Mark price: ", mark_price)
            limit_price = max(last_close, mark_price)
            print("Sell limit price:", limit_price)
            r.order_sell_crypto_limit(SYMBOL, quantity_held, limit_price)

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
        
    r.login(username=ACCOUNT, expiresIn=86400)
    print('Logged in')
    
    # Register signal interrupt behavior: log out
    signal.signal(signal.SIGINT, sigint_exit)

    trade_loop(config_file_path, experiment_dir_path)
    