import os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

DATA_ROOT_DIR = "./data/live"

def fetch_data_from_web():
    options = webdriver.ChromeOptions()
    options.headless = True
#     options.add_argument('--ignore-ssl-errors=yes')
#     options.add_argument('--ignore-certificate-errors')
    driver = webdriver.Chrome(executable_path='./chromedriver', options=options)
    
    driver.get('https://cointelegraph.com/bitcoin-price-index')
    
    export_menu = driver.find_element_by_css_selector('.export-main ul')
    print("Title: %s" % browser.title)
    

# wait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="cccChartDivBTC"]/div/div[2]/ul/li/ul/li[2]/ul/li[1]/a/span'))).click()

    driver.dispose()

if __name__ == "__main__":
    # Refetch data
    os.system("rm -rf " + DATA_ROOT_DIR)    
    os.system("mkdir " + DATA_ROOT_DIR)
    fetch_data_from_web()


# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options

# chromeOptions = Options()
# chromeOptions.binary_location = "/home/yuc016/bin/google-chrome"

# chromeOptions.add_argument('--headless')
# # chromeOptions.add_argument('--profile-directory=Default')
# # chrome_options.add_argument('--user-data-dir=~/.config/google-chrome')
# chromeOptions.add_argument("--no-sandbox");
# # chromeOptions.add_argument("--disable-dev-shm-usage");

# browser = webdriver.Chrome(executable_path="./chrome_headless/chromedriver", options=chromeOptions)

# browser.get("https://cointelegraph.com/bitcoin-price-index")

