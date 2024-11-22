# tests/test_data_processing.py

import unittest
from scripts.data_processing import preprocess_data
import pandas as pd

class TestDataProcessing(unittest.TestCase):
    def test_preprocess_data(self):
        df = pd.DataFrame({
            'text': ['Sample text data'],
            'label': ['Sample label']
        })
        X, y, vectorizer, label_encoder = preprocess_data(df)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)

if __name__ == '__main__':
    unittest.main()
