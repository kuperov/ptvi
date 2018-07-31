import ptvi
from autograd import numpy as np
from autograd.core import primitive

from elbo_grease.dist import mvn, invwishart
from elbo_grease.model import (Model, TransformedModel, list_all_elements,
                               list_lower_elements)


class DynamicLinearModel(Model):
    """
    The model is given by:

        ..math::
            z_t = Bz_{t-1} + Fx_{t-1} + \omega_{t-1}
            y_t = H_tz_t + Gx_t + \nu_t
            \omega_t\sim N(0,\Sigma_\omega)
            \nu_t\sim N(0,\Sigma_\nu)

    where:
        data:        (TxL) observed data
        B:           (KxK) dynamics matrix
        Sigma_omega: (KxK) innovation noise covariance
        H:           (LxK) transformation from latent to observed data
        Sigma_nu:    (LxL) observation error covariance
        z_0:         (K,)  mean of initial state
        Sigma_z_0:   (KxK) covariance of initial state

    Notation follows Lutkepohl, pp.626--
    """

    def __init__(self, obs_dim, state_dim):
        self.L = obs_dim
        self.K = state_dim

    def kalman_recursions(self, data, B, Sigma_omega, H, Sigma_nu, z_0,
                          Sigma_z_0):
        """Perform Kalman filter and smoother recursions.

        Args:
        Args:
            data:        observed data
            B:           dynamics matrix
            Sigma_omega: innovation noise covariance
            H:           transformation from latent to observed data
            Sigma_nu:    observation error covariance
            z_0:         mean of initial state
            Sigma_z_0:   covariance of initial state

          ..math::
            z_t = Bz_{t-1} + Fx_{t-1} + \omega_{t-1} \omega_t\sim N(0,\Sigma_\omega)
            y_t = H_tz_t + Gx_t + \nu_t,\quad        \nu_t\sim N(0,\Sigma_\nu)

        Notation follows Lutkepohl, pp.626--
        """
        T = data.shape[0]
        self.check_shapes(B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0, data)

        z_pred = np.zeros([T, self.K])  # z_{t|t-1}
        z_upd = np.zeros([T, self.K])  # z_{t|t}
        Sigma_z_pred = np.zeros([T, self.K, self.K])  # Sigma_{z_{t|t-1}}
        Sigma_z_upd = np.zeros([T, self.K, self.K])  # Sigma_{z_{t|t}}
        y_pred = np.zeros([T, self.L])  # y_{t|t-1}
        Sigma_y_pred = np.zeros([T, self.L, self.L])  # Sigma_{y_{t|t-1}}
        z_smooth = np.zeros([T, self.K])  # z_{t|T}
        Sigma_z_smooth = np.zeros([T, self.K, self.K])  # Sigma_{z_{t|T}}

        for t in range(1, T + 1):
            i = t - 1
            # prediction step
            if t == 1:
                z_pred[i] = B @ z_0  # + F@x[i]
                Sigma_z_pred[i] = B @ Sigma_z_0 @ B.T + Sigma_omega
            else:
                z_pred[i] = B @ z_upd[i - 1]  # + F@x[i]
                Sigma_z_pred[i] = B @ Sigma_z_upd[i - 1] @ B.T + Sigma_omega
            y_pred[i] = np.dot(H, z_pred[i])  # + G @ x[i]
            Sigma_y_pred[i] = H @ Sigma_z_pred[i] @ H.T + Sigma_nu

            # correction step
            gain = Sigma_z_pred[i] @ H.T @ np.linalg.inv(Sigma_y_pred[i])
            z_upd[i] = z_pred[i] + gain @ (data[i] - y_pred[i])
            Sigma_z_upd = Sigma_z_pred[i] - gain @ Sigma_y_pred @ gain.T

        # smoothing step
        z_smooth[T - 1] = z_upd[T - 1]
        Sigma_z_smooth[T - 1] = Sigma_z_smooth[T - 1]
        for t in range(T - 1, 0, -1):
            i = t - 1
            smooth = Sigma_z_upd[i] @ B.T @ np.linalg.inv(Sigma_z_pred[i])
            z_smooth[i] = (
                    z_upd[i]
                    + smooth @ (Sigma_z_pred[i + 1] - Sigma_z_smooth[
                i]) @ smooth.T
            )
            Sigma_z_smooth[i] = (
                    Sigma_z_upd[i]
                    - smooth @ (Sigma_z_pred[i + 1] - Sigma_z_smooth[
                i + 1]) @ smooth.T
            )
        return {
            'z_upd': z_upd, 'Sigma_z_upd': Sigma_z_upd, 'z_smooth': z_smooth,
            'Sigma_z_smooth': Sigma_z_smooth, 'y_pred': y_pred,
            'Sigma_y_pred': Sigma_y_pred
        }

    # TODO: initialize randomly, and accept rs as parameter
    def default_parameters(self):
        # Sigma_omega_chol = 0.01 * np.random.normal(size=[self.K, self.K])
        # Sigma_nu_chol = 0.01 * np.random.normal(size=[self.L, self.L])
        # return {
        #     'B': 0.1 * (np.random.uniform([self.K, self.K]) - 0.5),
        #     'Sigma_omega': Sigma_omega_chol @ Sigma_omega_chol.T,
        #     'H': 0.1 * (np.random.uniform([self.L, self.K]) - 0.5),
        #     'Sigma_nu': Sigma_nu_chol @ Sigma_nu_chol.T
        # }
        return {
            'B': np.zeros([self.K, self.K]),
            'Sigma_omega': np.eye(self.K),
            'H': 0.01 * np.ones([self.L, self.K]),
            'Sigma_nu': np.eye(self.L),
            'Sigma_z_0': np.eye(self.K),
            'z_0': np.zeros(self.K)
        }

    def log_lik(self, data, B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0):
        """Log likelihood of data given Kalman filter parameters.

        Args:
            data:        observed data
            B:           dynamics matrix
            Sigma_omega: innovation noise covariance
            H:           transformation from latent to observed data
            Sigma_nu:    observation error covariance
            z_0:         mean of initial state
            Sigma_z_0:   covariance of initial state

        The model is given by:

          ..math::
            z_t = Bz_{t-1} + Fx_{t-1} + \omega_{t-1} \omega_t\sim N(0,\Sigma_\omega)
            y_t = H_tz_t + Gx_t + \nu_t,\quad        \nu_t\sim N(0,\Sigma_\nu)

        Notation follows Lutkepohl, pp.626--
        """
        self.check_shapes(B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0, data)

        # we unroll first iteration of loop to set initial conditions
        # prediction step
        z_pred = np.dot(B, z_0)  # + F@x[i]
        Sigma_z_pred = np.dot(np.dot(B, Sigma_z_0), B.T) + Sigma_omega
        y_pred = np.dot(H, z_pred)  # + G @ x[i]
        Sigma_y_pred = np.dot(np.dot(H, Sigma_z_pred), H.T) + Sigma_nu

        # correction step
        gain = np.dot(np.dot(Sigma_z_pred, H.T), np.linalg.inv(Sigma_y_pred))
        z_upd = z_pred + np.dot(gain, (data[0] - y_pred))
        Sigma_z_upd = Sigma_z_pred - np.dot(np.dot(gain, Sigma_y_pred), gain.T)

        llik = mvn.logpdf(data[0], y_pred, Sigma_y_pred)

        for t in range(2, data.shape[0] + 1):
            i = t - 1
            # prediction step
            z_pred = np.dot(B, z_upd)  # + F @ x[i]
            Sigma_z_pred = np.dot(np.dot(B, Sigma_z_upd), B.T) + Sigma_omega
            y_pred = np.dot(H, z_pred)  # + G @ x[i]
            Sigma_y_pred = np.dot(np.dot(H, Sigma_z_pred), H.T) + Sigma_nu
            # correction step
            gain = np.dot(np.dot(Sigma_z_pred, H.T),
                          np.linalg.inv(Sigma_y_pred))
            z_upd = z_pred + np.dot(gain, (data[i] - y_pred))
            Sigma_z_upd = (
                    Sigma_z_pred - np.dot(np.dot(gain, Sigma_y_pred), gain.T))

            llik += mvn.logpdf(data[i], y_pred, Sigma_y_pred)
        return llik

    def log_prior(self, B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0):
        K = self.K
        L = self.L
        return (
            # mvn.logpdf(B.ravel(), np.zeros(self.K**2), 10*np.eye(self.K**2)) +
            # invwishart.logpdf(Sigma_omega, self.K+2, np.eye(self.K)) +
                mvn.logpdf(H.ravel(), np.zeros(K * L), 10 * np.eye(K * L)) +
                invwishart.logpdf(Sigma_nu, L + 2, np.eye(L)) +
                mvn.logpdf(z_0, np.zeros(K), np.eye(K)) +
                invwishart.logpdf(Sigma_z_0, K + 2, np.eye(K))
        )

    @primitive  # don't accumulate derivatives
    def simulate(self, size, B, Sigma_omega, H, Sigma_nu, z_0=None,
                 Sigma_z_0=None, rs=None):
        """Generate data as generated by the process described above.

        Runs in O(TKL) and is (obviously) not autograd differentiable.

        Returns:
          (Y, S) - tuple of observed data and state
        """
        rs = rs or np.random.RandomState()
        # promote to 2d matrices in case scalars (etc) were given
        # B = np.atleast_2d(B)
        # Sigma_omega = np.atleast_2d(Sigma_omega)
        # H = np.atleast_2d(H)
        # Sigma_nu = np.atleast_2d(Sigma_nu)
        S = np.zeros([size, self.K])
        Y = np.zeros([size, self.L])
        # initial state gets the steady-state variance if none given
        z_0 = np.zeros(self.K) if z_0 is None else z_0
        if Sigma_z_0 is None:
            Sigma_z_0 = self.steady_state_covariance(B, Sigma_omega)
        self.check_shapes(B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0)

        z0 = rs.multivariate_normal(z_0, Sigma_z_0)  # initial state
        S[0] = (np.dot(B, z0) +
                rs.multivariate_normal(np.zeros(self.K), Sigma_omega))
        Y[0] = (np.dot(H, S[0]) +
                rs.multivariate_normal(np.zeros(self.L), Sigma_nu))
        for t in range(2, size + 1):
            i = t - 1
            S[i] = (np.dot(B, S[i - 1]) +
                    rs.multivariate_normal(np.zeros(self.K), Sigma_omega))
            Y[i] = (np.dot(H, S[i]) +
                    rs.multivariate_normal(np.zeros(self.L), Sigma_nu))
        return Y, S

    def check_shapes(self, B, Sigma_omega, H, Sigma_nu, z_0, Sigma_z_0,
                     data=None):
        # Sig_o_c = np.array(Sigma_omega, copy=True)
        # assert np.all(np.linalg.eigvals(Sig_o_c) >= 0),\
        #     'Sigma_omega covariance matrix not positive semidefinite.'
        # Sig_n_c = np.array(Sigma_nu, copy=True)
        # assert np.all(np.linalg.eigvals(Sig_n_c) >= 0),\
        #     'Sigma_nu covariance matrix not positive semidefinite.'
        # Sig_z_c = np.array(Sigma_z_0, copy=True)
        # assert np.all(np.linalg.eigvals(Sig_z_c) >= 0),\
        #     'Sigma_z_0 covariance matrix not positive semidefinite.'
        assert B.shape == (self.K, self.K), 'B not (KxK) but {}'.format(B.shape)
        assert H.shape == (self.L, self.K), 'H not (LxK) but {}'.format(H.shape)
        assert z_0.shape == (self.K,), 'z_0 not K elements in length'
        assert Sigma_omega.shape == (self.K, self.K), \
            'Sigma_omega not KxK but {}'.format(Sigma_omega.shape)
        assert Sigma_nu.shape == (self.L, self.L), \
            'Sigma_nu not LxL but {}'.format(Sigma_nu.shape)
        # assert np.all(np.abs(np.linalg.eigvals(B)) < 1.),\
        #     'Eigenvalues of A should lie within the unit circle.'
        assert data is None or data.shape[1] == self.L, \
            'Data not L elements wide'

    def steady_state_covariance(self, B, Sigma_omega):
        assert np.all(np.linalg.eig(B)[0] < 1.)
        return np.dot(np.linalg.inv(np.eye(self.K ** 2) - np.kron(B, B)),
                      Sigma_omega.ravel()).reshape([self.K, self.K])

    # def plot_filtered_states(self, alpha=0.5):
    #     self.kalman_recursions()
    #     N, L = S.shape
    #     plt.figure()
    #     fig, axes = plt.subplots(nrows=1, ncols=L)
    #     if not isinstance(axes, np.ndarray):
    #         axes = np.array(
    #             axes)  # force an array when L=1 so the loop below works
    #     lower, upper = stats.norm().interval(alpha)
    #     for i, axis in enumerate(axes.ravel()):
    #         axis.plot(S[:, 0],
    #                   label='median, %0.f%% CI for $S^{(%d)}_{t|t}$' % (
    #                   100 * alpha, i),
    #                   color='green')
    #         sd = np.sqrt(P[:, i, i])
    #         axis.fill_between(range(1, N + 1), y1=S[:, 0] + sd * lower,
    #                           y2=S[:, 0] + sd * upper, alpha=0.4)
    #         axis.plot(unobs_S[:, 0], label='true $S^{(%d)}_t$' % i,
    #                   color='blue')
    #         axis.legend()
    #         axis.set_title('Filtered states $S^{(%d)}_{t|t}$' % i)
    #         axis.set_xlabel('t')

    # def plot_smoothed_states(S, P, alpha=0.5):
    #     N, L = S.shape
    #     plt.figure()
    #     fig, axes = plt.subplots(nrows=1, ncols=L)
    #     if not isinstance(axes, np.ndarray):
    #         axes = np.array(
    #             axes)  # force an array when L=1 so the loop below works
    #     lower, upper = stats.norm().interval(alpha)
    #     for i, axis in enumerate(axes.ravel()):
    #         axis.plot(S[:, 0],
    #                   label='median, %0.f%% CI for $S^{(%d)}_{t|T}$' % (
    #                   100 * alpha, i),
    #                   color='green')
    #         sd = np.sqrt(P[:, i, i])
    #         axis.fill_between(range(1, N + 1), y1=S[:, 0] + sd * lower,
    #                           y2=S[:, 0] + sd * upper, alpha=0.4)
    #         axis.plot(unobs_S[:, 0], label='true $S^{(%d)}_t$' % i,
    #                   color='blue')
    #         axis.legend()
    #         axis.set_title('Smoothed states $S^{(%d)}_{t|T}$' % i)
    #         axis.set_xlabel('t')


class StationaryLocalLevelModel(TransformedModel):
    """Stationary local level model, a simple case of a DLM.

    In this case we optimize over only B, H, and Sigma_nu, where Sigma_nu has
    been (generalized) Cholesky transformed to live in the unrestricted
    Eucledian space R**(KxK).
    """

    def __init__(self, obs_dim_L, state_dim_K):
        original_model = DynamicLinearModel(
            obs_dim=obs_dim_L, state_dim=state_dim_K)
        super().__init__(original_model)
        self.L = obs_dim_L
        self.K = state_dim_K

    def forward_transform(self, B, Sigma_omega, H, Sigma_nu, z_0=None,
                          Sigma_z_0=None):
        # silently initialize with steady-state values if none given
        z_0 = np.zeros(self.K) if z_0 is None else z_0
        if Sigma_z_0 is None:
            Sigma_z_0 = self.original_model.steady_state_covariance(
                B, Sigma_omega)
        # use steady-state innov variance and normalize innov noise
        return {
            'B': B, 'H': H, 'Sigma_nu_chol': np.linalg.cholesky(Sigma_nu),
            'z_0': z_0, 'Sigma_z_0_chol': np.linalg.cholesky(Sigma_z_0)
        }

    def inverse_transform(self, B, H, Sigma_nu_chol, z_0, Sigma_z_0_chol):
        """MUST be the exact inverse of `forward_transform`."""
        return {
            'B': B,
            'Sigma_omega': np.eye(self.K),  # normalize for identification
            'H': H,
            'Sigma_nu': np.dot(Sigma_nu_chol, Sigma_nu_chol.T),
            'z_0': z_0,
            'Sigma_z_0': np.dot(Sigma_z_0_chol, Sigma_z_0_chol.T)
        }

    def tabulate_variables(self):
        return (
                list_all_elements('B', (self.K, self.K)) +
                list_all_elements('H', (self.K, self.L)) +
                list_lower_elements('Sigma_nu', (self.L, self.L))
        )
