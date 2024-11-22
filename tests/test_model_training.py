# tests/test_model_training.py

import unittest
from scripts.model_training import SimpleTextClassifier
import torch

class TestModelTraining(unittest.TestCase):
    def test_model_forward(self):
        input_dim = 100
        model = SimpleTextClassifier(input_dim, 256, 128, 5)
        sample_input = torch.randn(1, input_dim)
        output = model(sample_input)
        self.assertEqual(output.shape[1], 5)

if __name__ == '__main__':
    unittest.main()
