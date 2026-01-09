import yfinance as yf
import pandas as pd

def test_fetch():
    try:
        tickers = yf.Tickers("ZC=F")
        df = tickers.download(period="1mo")
        print("Columns:", df.columns)
        print("Head:", df.head())
        
        # Simulate my logic
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        print("Columns after drop:", df.columns)
        
        # Check for OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                print(f"Found {col}")
            else:
                print(f"Missing {col}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch()
