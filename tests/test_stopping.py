import math
import unittest

from ptvi import (ExponentialStoppingHeuristic, NullStoppingHeuristic,
                  NoImprovementStoppingHeuristic)


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
        self.assertTrue(20 < i < 40-1, "Rule didn't fire in 1st quadrant")

    def test_no_improvement_stopping_heuristic(self):
        nish = NoImprovementStoppingHeuristic(skip=1, α=0.1)
        # test with a sine curve: increasing then decreasing; should stop in
        # shortly after pi/2 radians (20 steps)
        i = 0
        for i in range(40):
            elbo = math.sin(i * math.pi / 2 / 20)  # trace half circle (pi rad)
            if nish.early_stop(elbo):
                break
        self.assertTrue(20 < i < 40-1, "Rule didn't fire in 1st quadrant")
