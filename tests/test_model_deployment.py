# tests/test_model_deployment.py

import unittest
from scripts.model_deployment import ModelDeployment

class TestModelDeployment(unittest.TestCase):
    def test_load_model_artifacts(self):
        deployment = ModelDeployment('configs/config.yaml')
        self.assertIsNotNone(deployment.model)

if __name__ == '__main__':
    unittest.main()
