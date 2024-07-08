import yfinance as yf

def get_stock_info(stock_name = 'MSFT'):
    msft = yf.Ticker(stock_name)
    return msft.info

def get_stocks(stock_name = 'MSFT'):
    msft = yf.Ticker(stock_name)
    hist = msft.history(period="1mo")
    return hist

def get_identificators():
    return 'msft aapl goog'