import unittest
from app import app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_analyze(self):
        response = self.app.post('/analyze', json={"user_input": "Analyze TSLA stock"})
        self.assertEqual(response.status_code, 200)
        data =response.get_json()
        self.assertIn("stock", data)
        self.assertIn("rsi", data)
        self.assertIn("sentiment", data)
        self.assertIn("news_summary", data)
        self.assertIn("llm_decision", data)
        self.assertIn("llm_reasoning", data)

if __name__ == '__main__':
    unittest.main()