from warnings import warn
import unittest
from ptvi.priors import *
from torch.distributions import *


class TestPriors(unittest.TestCase):
    def test_normal_prior(self):
        np = NormalPrior(0, 2)
        self.assertEqual(str(np), "Normal(μ=0.0, σ²=4.0)")
        d = np.to_distribution()
        self.assertIsInstance(d, Normal)
        self.assertEqual(d.mean.dtype, torch.float32)
        dlong = np.to_distribution(dtype=torch.float64)
        self.assertEqual(dlong.mean.dtype, torch.float64)
        # the next line will fail with operation error if wrong type used
        lp = dlong.log_prob(torch.tensor(0.5, dtype=torch.float64))

    def test_normal_prior_cuda(self):
        if not torch.cuda.is_available():
            warn("Skipping CUDA test")
            return
        np = NormalPrior(0, 1)
        cuda = torch.device("cuda")
        dcuda = np.to_distribution(device=cuda)
        self.assertEqual(dcuda.mean.device.type, "cuda")

    def test_log_normal_prior(self):
        lnp = LogNormalPrior(0, 1)
        self.assertEqual(str(lnp), "LogNormal(μ=0.0, σ²=1.0)")
        self.assertIsInstance(lnp.to_distribution(), LogNormal)

    def test_beta_prior(self):
        bp = BetaPrior(2, 3)
        self.assertEqual(str(bp), "Beta(α=2.0, β=3.0)")
        self.assertIsInstance(bp.to_distribution(), Beta)

    def test_modified_beta(self):
        mbp = ModifiedBetaPrior(1, 1)
        self.assertEqual(str(mbp), '2*Beta(α=1.0, β=1.0)-1')
        # should be the same as uniform (-1, 1)
        d = mbp.to_distribution()
        self.assertTrue(torch.allclose(torch.tensor([0.5,0.5,0.5]),
            torch.exp(d.log_prob(torch.tensor([-0.9,0.,0.9])))))

    def test_chisquare(self):
        chsq = Chi2Prior(df=1)
        self.assertEqual(str(chsq), 'χ²(df=1)')
        d = chsq.to_distribution()
        self.assertIsInstance(d, torch.distributions.Chi2)
