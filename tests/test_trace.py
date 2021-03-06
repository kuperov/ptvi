import unittest
from unittest.mock import patch
import numpy as np
import torch

from ptvi import (
    PointEstimateTracer,
    UnivariateGaussian,
    FilteredSVModelDualOpt,
    DualPointEstimateTracer,
)


class TestPointEstimateTracer(unittest.TestCase):
    def test_trace(self):
        m = UnivariateGaussian()
        t = PointEstimateTracer(m)
        for i in range(10):
            t.append(torch.tensor([i + 1., i + 2.]), 100. + i)
        u_arr = t.to_unconstrained_array()
        c_arr = t.to_constrained_array()
        self.assertEqual(u_arr.shape, (10, 2))
        self.assertEqual(c_arr.shape, (10, 2))
        self.assertTrue(np.allclose(c_arr[:, 0], u_arr[:, 0]))
        self.assertTrue(np.allclose(np.log(c_arr[:, 1]), u_arr[:, 1]))
        patch("ptvi.model.plt.show", t.plot(figsize=[8, 10], true={"μ": 0., "σ": 1.}))


class TestDualPointEstimateTracer(unittest.TestCase):
    def test_trace(self):
        m = FilteredSVModelDualOpt(input_length=10)
        t = DualPointEstimateTracer(m)
        for i in range(10):
            ζ, η = torch.randn(3), torch.randn(2)
            t.append(ζ, η, 100. + i)
        u_arr = t.to_unconstrained_array()
        c_arr = t.to_constrained_array()
        self.assertEqual(u_arr.shape, (10, 5))
        self.assertEqual(c_arr.shape, (10, 5))
        patch("ptvi.model.plt.show", t.plot(figsize=[8, 10]))
