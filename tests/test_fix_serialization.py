import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import json
import sys
import os

# Add project root to path to import main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import get_corn_futures

class TestSerialization(unittest.TestCase):
    @patch('main.yf.Tickers')
    def test_get_corn_futures_serialization(self, mock_tickers):
        # Mock yfinance returning a MultiIndex DataFrame (simulate the issue)
        dates = pd.date_range(start='2025-01-01', periods=5)
        # Create a MultiIndex DataFrame: levels (Price, Ticker)
        columns = pd.MultiIndex.from_product([['Close', 'Open'], ['ZC=F']], names=['Price', 'Ticker'])
        data = [[4.5, 4.4], [4.6, 4.5], [4.7, 4.6], [4.5, 4.4], [4.8, 4.7]]
        df = pd.DataFrame(data, index=dates, columns=columns)
        
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.download.return_value = df
        mock_tickers.return_value = mock_ticker_instance

        # Call the function
        result = get_corn_futures('2025-01-01', '2025-01-10')

        # Verify result is a list of dicts
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        
        # Verify JSON serialization
        try:
            json_str = json.dumps(result)
            self.assertIsInstance(json_str, str)
        except TypeError as e:
            self.fail(f"Result is not JSON serializable: {e}")

        # Verify content
        first_item = result[0]
        self.assertIn('date', first_item)
        self.assertIn('close', first_item)
        self.assertIsInstance(first_item['close'], float)
        self.assertEqual(first_item['close'], 4.5)

    @patch('main.yf.Tickers')
    def test_get_corn_futures_single_index(self, mock_tickers):
        # Mock yfinance returning a simple DataFrame (older versions)
        dates = pd.date_range(start='2025-01-01', periods=3)
        data = {'Close': [4.5, 4.6, 4.7], 'Open': [4.4, 4.5, 4.6]}
        df = pd.DataFrame(data, index=dates)
        
        # Setup mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.download.return_value = df
        mock_tickers.return_value = mock_ticker_instance

        # Call the function
        result = get_corn_futures('2025-01-01', '2025-01-05')

        # Verify JSON serialization
        try:
            json.dumps(result)
        except TypeError as e:
            self.fail(f"Result is not JSON serializable: {e}")

if __name__ == '__main__':
    unittest.main()
