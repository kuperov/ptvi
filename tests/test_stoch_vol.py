import unittest
from ptvi import StochVolModel, VITimeSeriesResult


class TestStochVolModel(unittest.TestCase):

    def test_simulate(self):
        m = StochVolModel(input_length=50)
        y, b = m.simulate(λ=0.5, σ=0.1, φ=0.95)
        self.assertEqual(y.shape, (50,))
        self.assertEqual(b.shape, (50,))

    def test_training_loop(self):
        m = StochVolModel(input_length=50, quiet=True)
        y, b = m.simulate(λ=0.5, σ=0.5, φ=0.95)
        fit = m.training_loop(y, max_iters=5)
        self.assertIsInstance(fit, VITimeSeriesResult)
