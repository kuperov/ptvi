import unittest
from ptvi import StochVolModel, sgvb, MVNPosterior


class TestStochVolModel(unittest.TestCase):
    def test_simulate(self):
        m = StochVolModel(input_length=50)
        y, b = m.simulate(λ=0.5, σ=0.1, φ=0.95)
        self.assertEqual(y.shape, (50,))
        self.assertEqual(b.shape, (50,))

    def test_training_loop(self):
        m = StochVolModel(input_length=50)
        y, b = m.simulate(λ=0.5, σ=0.5, φ=0.95)
        fit = sgvb(m, y, max_iters=5, quiet=True)
        self.assertIsInstance(fit, MVNPosterior)
