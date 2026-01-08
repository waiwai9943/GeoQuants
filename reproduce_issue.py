import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_corn_futures(start_date, end_date):
    """Fetches Corn Futures (ZC=F) data from Yahoo Finance."""
    try:
        # ZC=F is the ticker for Corn Futures on Yahoo Finance
        tickers = yf.Tickers("ZC=F")
        df = tickers.download(start=start_date, end=end_date)
        
        print("Columns:", df.columns)
        print("Head:", df.head())
        
        if df.empty:
            return []

        df = df.reset_index()
        # Check if Date is in columns or index
        print("Columns after reset_index:", df.columns)
        
        if 'Date' in df.columns:
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        else:
             # It might be lower case 'date' or something else if index name was different
             print("Date column not found, checking index name")
             # Try finding the date column
             for col in df.columns:
                 if pd.api.types.is_datetime64_any_dtype(df[col]):
                     df['date'] = df[col].dt.strftime('%Y-%m-%d')
                     break

        if 'Close' in df.columns:
            df['close'] = df['Close'].round(2)
        else:
             # Handle MultiIndex or different case
             print("Close column not found directly.")
             # If MultiIndex, it might be ('Close', 'ZC=F')
             try:
                 df['close'] = df['Close']['ZC=F'].round(2)
             except:
                 pass

        print("Types before to_dict:")
        print(df[['date', 'close']].dtypes)
        
        result = df[['date', 'close']].to_dict('records')
        print("First record:", result[0])
        return result

    except Exception as e:
        print(f"Yahoo Finance Error: {e}")
        import traceback
        traceback.print_exc()
        return []

# Test with recent dates
end = datetime.now()
start = end - timedelta(days=30)
start_str = start.strftime('%Y-%m-%d')
end_str = end.strftime('%Y-%m-%d')

print(f"Fetching from {start_str} to {end_str}")
data = get_corn_futures(start_str, end_str)
print("Result type:", type(data))
if data:
    print("First item type:", type(data[0]))
    print("First item values types:", {k: type(v) for k, v in data[0].items()})
