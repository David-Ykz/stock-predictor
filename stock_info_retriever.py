import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
TICKER = 'MSFT'

def getStockData(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_KEY}'
    stockData = requests.get(url).json()

    with open(f"{ticker}.json", "w") as json_file:
        json.dump(stockData, json_file, indent=4)

    print(f"saved data to {ticker}.json")

def readToDf(filePath):
    with open(filePath, "r") as file:
        stockData = json.load(file)

    timeSeries = stockData.get("Time Series (Daily)", {})
    
    data = []
    
    for date, daily_data in timeSeries.items():
        row = {
            "Date": date,
            "Open": float(daily_data["1. open"]),
            "High": float(daily_data["2. high"]),
            "Low": float(daily_data["3. low"]),
            "Close": float(daily_data["4. close"]),
            "Volume": int(daily_data["5. volume"])
        }
        data.append(row)    
    df = pd.DataFrame(data) 
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df.sort_values(by='Date', ascending=True)

#getStockData("AAPL")
#print(readToDf("IBM.json"))

