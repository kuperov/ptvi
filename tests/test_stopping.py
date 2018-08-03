import math
import unittest
import torch

from ptvi import (
    ExponentialStoppingHeuristic,
    NullStoppingHeuristic,
    NoImprovementStoppingHeuristic,
    MedianGrowthStoppingHeuristic,
    SupGrowthStoppingHeuristic,
)


class TestEarlyStoppingRules(unittest.TestCase):
    def test_null_stopping_heuristic(self):
        nsr = NullStoppingHeuristic()
        for i in range(20, 1, -1):
            self.assertFalse(nsr.early_stop(float(i)))

    def test_exponential_stopping_heuristic(self):
        esh = ExponentialStoppingHeuristic(N=1, M=5, α=0.1)
        # test with a sine curve: increasing then decreasing; should stop in
        # shortly after pi/2 radians (20 steps)
        i = 0
        for i in range(40):
            elbo = math.sin(i * math.pi / 2 / 20)  # trace half circle (pi rad)
            if esh.early_stop(elbo):
                break
        self.assertTrue(20 < i < 40 - 1, "Rule didn't fire in 1st quadrant")

    def test_no_improvement_stopping_heuristic(self):
        nish = NoImprovementStoppingHeuristic(skip=1, α=0.1, min_steps=0)
        # test with a sine curve: increasing then decreasing; should stop in
        # shortly after pi/2 radians (20 steps)
        i = 0
        for i in range(40):
            elbo = math.sin(i * math.pi / 40)  # trace half circle (pi rad)
            if nish.early_stop(elbo):
                break
        self.assertTrue(20 < i < 40 - 1, "Rule didn't fire in 1st quadrant")

    def test_minimum_median_improvement_rate(self):
        length, initial = 200, -100.

        def stop_with_drift(drift):
            elbos = torch.cumsum(torch.randn((length)) + drift, 0)
            h = MedianGrowthStoppingHeuristic(skip=1, patience=10, ε=0.1, min_steps=50)
            for i in range(length):
                if h.early_stop(elbos[i]):
                    return i
            return -1

        torch.manual_seed(123)
        self.assertTrue(stop_with_drift(10. / 10.) == -1)
        self.assertTrue(stop_with_drift(0.5 / 10.) > 50)

    def test_sup_growth(self):
        heur = SupGrowthStoppingHeuristic(
            patience=40, skip=1, min_steps=50, ε=1e-1, α=.1
        )
        i = 0
        for i in range(200):
            # period = 20 < patience = 40, so should stop early
            elbo = math.sin(i * math.pi * 2 / 20) + (1e-1 / 20 * i)
            if heur.early_stop(elbo):
                break
        self.assertEqual(i, 199)
        # period = 400 << patience = 40, so should not stop
        heur = SupGrowthStoppingHeuristic(
            patience=40, skip=1, min_steps=50, ε=1e-1, α=.1
        )
        for i in range(200):
            # "elbo" continually declining
            elbo = math.cos(i * math.pi * 2 / 400) + (1e-1 / 20 * i)
            if heur.early_stop(elbo):
                break
        self.assertLess(i, 199)
