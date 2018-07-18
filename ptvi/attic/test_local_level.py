import unittest
from ptvi.attic.local_level import UnivariateLocalLevel
from autograd import numpy as np, jacobian


class TestUnivariateLocalLevel(unittest.TestCase):

    def setUp(self):
        self.model = UnivariateLocalLevel(τ=50)

    def test_T_Tinv(self):
        ς, φ = self.model.T(σ=15., ρ=0.5)
        σ, ρ = self.model.Tinv(ς, φ)
        self.assertAlmostEqual(σ, 15.)
        self.assertAlmostEqual(ρ, 0.5)

    def test_simulate_compute_joint(self):
        import numbers
        rs = np.random.RandomState(seed=123)
        z, y = self.model.simulate(γ=100., η=2., σ=1.5, ρ=0.8, rs=rs)
        ll_at_true = self.model.log_joint(y=y, z=z, γ=100., η=2., σ=1.5, ρ=0.8)
        ll_alt = self.model.log_joint(y=y, z=z, γ=200., η=-2., σ=0.5, ρ=-0.8)
        self.assertIsInstance(ll_at_true, numbers.Number)
        self.assertGreater(ll_at_true, ll_alt)
        ς, φ = self.model.T(σ=1.5, ρ=0.8)
        ζ = np.concatenate([z, [100., 2., ς, φ]])
        ll_true_tfm = self.model.transformed_log_joint(y=y, ζ=ζ)
        # includes jacobian determinant, so should *not* be equal to ll_at_true
        self.assertIsInstance(ll_true_tfm, numbers.Number)
        self.assertNotEqual(ll_true_tfm, ll_at_true)
        # now smoke test a bunch of random draws...
        k = len(ζ)
        djoint_dζ = jacobian(self.model.transformed_log_joint, argnum=1)
        for i in range(5):
            ζ = rs.normal(size=k)
            # should always return finite values
            ll = self.model.transformed_log_joint(y=y, ζ=ζ)
            self.assertTrue(np.all(np.isfinite(ll)), msg=f'{i}. ll not finite')
            # same for the gradient
            dpdζ = djoint_dζ(y, ζ)
            self.assertTrue(np.all(np.isfinite(dpdζ)), msg=f'{i}. grad inf.')

    def test_smoke_test_advi(self):
        rs = np.random.RandomState(seed=123)
        y, _ = self.model.simulate(γ=100., η=2., σ=1.5, ρ=0.8, rs=rs)
        μ, L = self.model.advi(y, n_draws=5, noisy=True)
        sd = np.sqrt(np.trace(L@L.T))
        result = pd.DataFrame({'mean': μ, 'sd': sd})
        print(result)
