import unittest
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import app, SESSION_FILE

class TestSessionManagement(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # Clean up session file before test
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

    def tearDown(self):
        # Clean up session file after test
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

    def test_save_and_load_session(self):
        # Test Data
        session_data = {
            "polygon": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 0]]]
            },
            "startDate": "2024-01-01",
            "endDate": "2024-02-01",
            "indexType": "NDVI",
            "chartData": {"dummy": "data"}
        }

        # 1. Test Save
        response_save = self.app.post('/save_session', 
                                      data=json.dumps(session_data),
                                      content_type='application/json')
        self.assertEqual(response_save.status_code, 200)
        self.assertIn("Session saved", str(response_save.data))

        # Verify file exists
        self.assertTrue(os.path.exists(SESSION_FILE))

        # 2. Test Load
        response_load = self.app.get('/load_session')
        self.assertEqual(response_load.status_code, 200)
        
        loaded_data = json.loads(response_load.data)
        self.assertEqual(loaded_data['startDate'], "2024-01-01")
        self.assertEqual(loaded_data['chartData']['dummy'], "data")

    def test_load_nonexistent_session(self):
        response = self.app.get('/load_session')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), {})

if __name__ == '__main__':
    unittest.main()
