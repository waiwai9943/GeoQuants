import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import get_corn_futures

class CornFuturesTest(unittest.TestCase):
    @patch('main.yf.Tickers')
    def test_get_corn_futures(self, mock_tickers):
        # Create a mock DataFrame
        mock_data = {
            'Close': [450.0, 452.5]
        }
        mock_index = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        mock_df = pd.DataFrame(mock_data, index=mock_index)

        # Configure the mock to return the mock DataFrame
        mock_instance = MagicMock()
        mock_instance.download.return_value = mock_df
        mock_tickers.return_value = mock_instance

        # Call the function with test dates
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        result = get_corn_futures(start_date, end_date)

        # Assert the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['date'], '2024-01-01')
        self.assertEqual(result[0]['close'], 450.0)
        self.assertEqual(result[1]['date'], '2024-01-02')
        self.assertEqual(result[1]['close'], 452.5)
        
        # Assert that yf.Tickers was called with the correct ticker
        mock_tickers.assert_called_with("ZC=F")
        
        # Assert that download was called with the correct dates
        mock_instance.download.assert_called_with(start=start_date, end=end_date)

if __name__ == '__main__':
    unittest.main()
