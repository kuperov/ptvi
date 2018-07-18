import unittest
from ptvi.attic.svm import SVM
from ptvi.util import *
from autograd import numpy as np, jacobian


class TestSVM(unittest.TestCase):

    def setUp(self):
        np.seterr(all='raise')
        # The prior has been modified to reduce the chances of drawing params
        # that have unstable dynamics within a few draws. That's still possible
        # anyway, so we are also strict with setting the random seed to a value
        # that keeps us out of trouble.
        self.model = SVM(μ0=0., Vμ=.5, φ0=.9, Vφ=.5**2,
            γ0=np.zeros(2), Ω0=0.1*np.eye(2),
            β0=0., Vβ=.05, νσ2=10, Sσ2=.2**2*(10-1),
            νΩ=10, SΩ=(10+3)*np.diag([.1**2, .25**2]))

    def test_simulate_prior_respects_random_state(self):
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        θ0 = self.model.simulate_prior(rs=rs)
        for _ in range(10):
            rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
            θ = self.model.simulate_prior(rs=rs)
            for k in θ0:
                self.assertTrue(np.all(θ0[k] == θ[k]))

    def test_simulate_respects_random_state(self):
        T = 50
        θ0 = {'μ': 0.01, 'φ':  0.98, 'β': 0.51, 'σ2': 0.05,
              'Ω': np.array([[0.015, 0.018], [0.018,  0.043]])}
        rs = np.random.RandomState(seed=123)
        data0 = self.model.simulate(T, **θ0, rs=rs)
        y0, h0, γ0 = data0['y'], data0['h'], data0['γ']
        for _ in range(10):
            rs = np.random.RandomState(seed=123)
            data = self.model.simulate(T, **θ0, rs=rs)
            y, h, γ = data['y'], data['h'], data['γ']
            self.assertTrue(np.all(y0 == y))
            self.assertTrue(np.all(h0 == h))
            self.assertTrue(np.all(γ0 == γ))

    def test_T(self):
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        ψ = self.model.simulate_prior(rs=rs)
        data = self.model.simulate(50, **ψ)
        ζ = ψ.copy()
        ζ['η'] = np.arctanh(ζ['φ'])
        ζ['ln_σ2'] = np.log(ζ['σ2'])
        ζ['ω'] = pd_to_vec(ζ['Ω'])
        del ζ['φ']; del ζ['σ2']; del ζ['Ω']
        θ = self.model.pack_ζ(data['γ'], data['h'], **ζ)
        ζ2 = self.model.unpack_ζ(θ)
        for key in ζ:
            self.assertTrue(np.all(ζ[key] == ζ2[key]),
                            msg='{} != {}'.format(ζ[key], ζ2[key]))
        self.assertTrue(np.all(data['h'] == ζ2['h']))
        self.assertTrue(np.all(data['γ'] == ζ2['γ']))

    def test_simulate_and_joint(self):
        # draw parameters, simulate data, compute log joint density and gradient
        import numbers
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        T = 50
        ψ = self.model.simulate_prior(rs=rs)
        self.assertTrue(list(ψ.keys()) == ['μ', 'φ', 'β', 'σ2', 'Ω'])
        data = self.model.simulate(T, **ψ)
        self.assertEqual(data['y'].shape, (T,))
        self.assertEqual(data['h'].shape, (T,))
        self.assertEqual(data['γ'].shape, (T, 2))
        ll = self.model.log_p(**data, **ψ)
        self.assertIsInstance(ll, numbers.Number)

        # check the Jacobian term isn't missing
        ζ = self.model.T(data['γ'], data['h'], **ψ)
        ll_unrestr = self.model.log_p_unrestricted(data['y'], **ζ)
        self.assertNotEqual(ll, ll_unrestr)

        # 'packed' version of log_p takes a vector
        θ = self.model.pack_ζ(**ζ)
        self.assertEqual(ll_unrestr, self.model.log_p_packed(data['y'], θ))

        # packed version should be differentiable
        log_p_grad = jacobian(self.model.log_p_packed, argnum=1)
        gr = log_p_grad(data['y'], θ)
        self.assertTrue(np.all(np.isfinite(gr)))

    def test_elbo_hat(self):
        # should be finite and differentiable
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        ψ = self.model.simulate_prior(rs=rs)
        T = 10; k = 3*T+7
        data = self.model.simulate(T, **ψ)
        φ0 = np.concatenate([[0]*k, tril_to_vec(np.eye(k))])
        ehat = self.model.elbo_hat(data['y'], φ0)
        self.assertTrue(np.isfinite(ehat))
        ehat_grad = jacobian(self.model.elbo_hat, argnum=1)(data['y'], φ0)
        self.assertTrue(np.all(np.isfinite(ehat_grad)))
        # should see one gradient per element of phi, i.e. k+k(k+1)/2
        self.assertEqual(k+k*(k+1)//2, len(ehat_grad))

    def test_jacobian(self):
        # smoke test for sgvb
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        τ = 5
        k = 3 * τ + 7
        ψ = self.model.simulate_prior(rs=rs)
        φ0 = np.concatenate([[0]*k, tril_to_vec(np.eye(k))])
        data = self.model.simulate(τ, **ψ, rs=rs)
        ζ = self.model.T(data['γ'], data['h'], **ψ)
        # # smoke test the transform itself
        # θ = self.model.Tinv(**ζ)
        det = self.model.log_det_J_Tinv(ζ['η'], ζ['ln_σ2'], ζ['ω'])
        self.assertTrue(np.isfinite(det))

    def test_elbo_hat_gradient(self):
        rs = np.random.RandomState(seed=123)
        ψ = self.model.simulate_prior(rs=rs)
        τ = 5
        k = 3 * τ + 7
        data = self.model.simulate(τ, **ψ, rs=rs)
        φ = np.concatenate([[0] * k, tril_to_vec(np.eye(k))])
        η = rs.normal(size=k)
        ehat = lambda φ_: self.model.elbo_hat(data['y'], φ_, η)
        ε = 1e-8
        ε_at = lambda ι: np.concatenate([
            [0]*ι, [ε], [0]*max(0, len(φ) - ι - 1)])
        g = np.zeros(len(φ))
        for ι in range(len(φ)):
            ε_ι = ε_at(ι)
            g[ι] = (ehat(φ + ε_ι) - ehat(φ - ε_ι))/(2 * ε)
        jac = jacobian(ehat)(φ)
        self.assertTrue(np.allclose(jac, g, atol=1e-5))

    def test_sgvb(self):
        # smoke test for sgvb
        rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
        ψ = self.model.simulate_prior(rs=rs)
        τ = 5
        k = 3 * τ + 7
        data = self.model.simulate(τ, **ψ, rs=rs)
        results = self.model.sgvb(data['y'], n_draws=100, max_iter=20, noisy=True, rs=rs)
        self.assertIsInstance(results, dict)
        self.assertEqual(np.shape(results['μ']), (k,))
        self.assertEqual(np.shape(results['L']), (k, k))

    # def test_gradients(self):
    #     rs = np.random.RandomState(seed=123)  # cherry-picked stable seed
    #     ψ = self.model.simulate_prior(rs=rs)
    #     τ = 5
    #     k = 3 * τ + 7
    #     data = self.model.simulate(τ, **ψ, rs=rs)
    #     check_grads(self.model.T, modes=['rev'])()
