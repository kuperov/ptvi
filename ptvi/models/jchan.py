from time import time
import pickle
import pandas as pd
import numpy as np
from numpy.linalg import inv, solve
from scipy.sparse import (
    spdiags,
    block_diag,
    kron as spkron,
    identity as speye,
    csc_matrix,
    bmat as spbmat,
)
from scipy.sparse.linalg import spsolve
from scipy import sparse, stats, special
import matplotlib.pyplot as plt
import os


########################### hackity hack #############################
##################### on linux, use scikit.sparse ####################
def spchol(A):
    return sparse.csc_matrix(np.linalg.cholesky(A.todense()))


########################### hackity hack #############################


def svm_sampler(
    y, nloop=55_000, burnin=5_000, is_dym_prob=False, is_alp_const=False, rs=None
):
    """Sample from unobserved components model with stoch vol in mean.

    Chan, Joshua CC. "The stochastic volatility in mean model with time-varying
    parameters: An application to inflation modeling." Journal of Business &
    Economic Statistics 35.1 (2017): 17-28.

    The model is as follows, where B denotes the backshift operator, εₜ and νₜ
    are iid standard normal, ωₜ is iid multivariate standard normal, and Ω^½
    denotes the cholesky factor of Ω.

      ..math::
        yₜ = τₜ + αₜ exp(hₜ) + exp(½hₜ)εₜ
        hₜ = μ + φ(Βhₜ - μ) + βΒyₜ+ σνₜ
        γₜ = (αₜ, τₜ)ᵀ
        γₜ = Βγₜ + Ω^½ᵀ ωₜ
        τ₁ ~ N(0, Vτ)
        h₁ ~ N(μ, σ²/(1 - φ²))

    Args:
        nloop:       number of samples to draw
        burnin:      length of warm-up sample
        is_dym_prob: if True, compute dynamic probabilities
        is_α_const:  if True, α is restricted to be constant

    """
    assert isinstance(y, pd.core.series.Series) or isinstance(y, np.ndarray)
    rs = rs or np.random.RandomState()
    y = np.array(y).squeeze()
    T = y.size
    tid = np.linspace(1948, 2013.25, T)  # srsly bro
    assert (
        not is_alp_const or is_dym_prob
    ), "For UC-SVM-const dynamic probabilities cannot be computed"

    # prior
    phi0 = .97
    Vphi = .1 ** 2
    mu0 = 0
    Vmu = 10
    Vgam = 10 * np.eye(2)
    invVgam = np.linalg.inv(Vgam)
    beta0 = 0
    Vbeta = 10
    nuOmega = 10
    if is_alp_const:
        SOmega = (nuOmega + 3) * np.diag([1e-12, .25 ** 2])
        model_name = "UC-SVM-const"
    else:
        SOmega = (nuOmega + 3) * np.diag([1e-2, .25 ** 2])
        model_name = "UC-SVM"
    nuh = 10
    Sh = .2 ** 2 * (nuh - 1)

    # initialize the Markov chain
    beta = 0.
    mu = np.log(np.var(y, ddof=1))
    phi, sig2 = .98, 0.2 ** 2
    Omega = np.diag([.1 ** 2, .25 ** 2])
    h = mu + np.sqrt(sig2) * rs.normal(size=T)
    exph = np.exp(h)

    # initialize storage
    store_theta = np.zeros([nloop - burnin, 7])  # [mu beta phi sig2 Omega([1 2 4])]
    store_tau = np.zeros([nloop - burnin, T])
    store_alp = np.zeros([nloop - burnin, T])
    store_h = np.zeros([nloop - burnin, T])

    # compute a few things outside the loop
    H = spdiags(
        [np.ones([2 * T]), -np.ones([2 * T])], [0, -2], m=2 * T, n=2 * T, format="csr"
    )
    Hphi = spdiags([np.ones([T]), -phi * np.ones([T])], [0, -1], m=T, n=T, format="csr")
    Hphi_off_diag_idxs = (np.arange(1, T), np.arange(T - 1))
    Xgam = block_diag(np.transpose([exph, np.ones(T)]), format="csr")
    Xgam_phi_idxs = (np.arange(T), np.arange(0, 2 * T, 2))
    newnuh = T / 2 + nuh
    counth = 0
    countlam = 0
    print("Starting MCMC for {}...".format(model_name))
    start_time = time()
    invOmegagamB1 = csc_matrix((2, 2 * (T - 1)))
    invOmegagamB2 = csc_matrix((2 * (T - 1), 2))
    for loop in range(1, nloop + 1):
        # sample gam
        invOmegagam = spbmat(
            [
                [invVgam, invOmegagamB1],
                [invOmegagamB2, spkron(speye(T - 1), inv(Omega))],
            ]
        )
        Xgam[Xgam_phi_idxs] = exph
        temp1 = Xgam.T @ spdiags(1. / exph, 0, T, T)
        Kgam = H.T @ invOmegagam @ H + temp1 @ Xgam
        Cgam = spchol(Kgam)
        gamhat = spsolve(Kgam, temp1 @ y)
        gam = gamhat + spsolve(Cgam.T, rs.normal(size=2 * T))
        alp = gam[np.arange(0, T * 2, 2)]
        tau = gam[np.arange(1, T * 2, 2)]

        # sample h
        HinvSH = (
            Hphi.T
            @ spdiags(np.concatenate([[1 - phi ** 2], np.ones(T - 1)]) / sig2, 0, T, T)
            @ Hphi
        )
        deltah = spsolve(
            Hphi,
            np.concatenate([[mu], mu * (1 - phi) * np.ones(T - 1) + y[:-1] * beta]),
        )
        HinvSHdeltah = HinvSH @ deltah
        s2 = (y - tau) ** 2
        errh = 1
        ht = h  # htilde
        while errh > 1e-3:  # newton-raphson method
            expht = np.exp(ht)
            sinvexpht = s2 / expht
            alp2expht = alp ** 2 * expht
            fh = -0.5 + 0.5 * sinvexpht - 0.5 * alp2expht
            Gh = 0.5 * sinvexpht + 0.5 * alp2expht
            Kh = HinvSH + spdiags(Gh, 0, T, T)
            newht = spsolve(Kh, fh + Gh * ht + HinvSHdeltah)
            errh = np.max(np.abs(newht - ht))
            ht = newht
        cholHh = spchol(Kh)

        # AR-step:
        hstar = ht
        uh = hstar - deltah
        logc = (
            -0.5 * uh.T @ HinvSH @ uh
            - 0.5 * np.sum(hstar)
            - 0.5 * np.exp(-hstar).T @ (y - tau - alp * np.exp(hstar)) ** 2
            + np.log(3)
        )
        flag = 0
        while not flag:
            hc = ht + spsolve(cholHh.T, rs.normal(size=T))
            vhc = hc - ht
            uhc = hc - deltah
            alpARc = (
                -0.5 * uhc.T @ HinvSH @ uhc
                - 0.5 * np.sum(hc)
                - 0.5 * np.exp(-hc).T @ (y - tau - alp * np.exp(hc)) ** 2
                + 0.5 * vhc.T @ Kh @ vhc
                - logc
            )
            if alpARc > np.log(rs.uniform()):
                flag = 1
        # MH-step
        vh = h - ht
        uh = h - deltah
        alpAR = (
            -0.5 * uh.T @ HinvSH @ uh
            - 0.5 * np.sum(h)
            - 0.5 * np.exp(-h).T @ (y - tau - alp * np.exp(h)) ** 2
            + 0.5 * vh.T @ Kh @ vh
            - logc
        )
        if alpAR < 0:
            alpMH = 1
        elif alpARc < 0:
            alpMH = -alpAR
        else:
            alpMH = alpARc - alpAR
        if alpMH > np.log(rs.uniform()) or loop == 1:
            h = hc
            exph = np.exp(h)
            counth = counth + 1

        # sample beta
        ybeta = h[1:] - mu - phi * (h[:-1] - mu)
        Dbeta = 1. / (1. / Vbeta + np.dot(y[:-1], y[:-1]) / sig2)
        betahat = Dbeta * (beta0 / Vbeta + np.dot(y[:-1], ybeta) / sig2)
        beta = betahat + np.sqrt(Dbeta) * rs.normal()

        # sample Omega
        err = (gam[2:] - gam[:-2]).reshape(T - 1, 2)
        Omega = stats.invwishart.rvs(
            scale=SOmega + err.T @ err, df=nuOmega + T - 1, random_state=rs
        )

        # sample sig2
        errh = np.concatenate(
            [
                [(h[0] - mu) * np.sqrt(1 - phi ** 2)],
                h[1:] - phi * h[:-1] - mu * (1 - phi) - y[:-1] * beta,
            ]
        )
        newSh = Sh + 0.5 * np.sum(errh ** 2)
        sig2 = 1. / stats.gamma.rvs(a=newnuh, scale=1. / newSh, random_state=rs)
        # ... stats.invgamma.rvs ... scale=newSh ?

        # sample mu and phi jointly
        def flam(x):
            return (
                -(1 - x[1]) ** 2 / (2 * sig2) * (h[0] - x[0]) ** 2
                - 1
                / (2 * sig2)
                * np.sum((h[1:] - x[1] * h[:-1] - x[0] * (1 - x[1])) ** 2)
                - 1 / (2 * Vphi) * (x[1] - phi0) ** 2
                - 1 / (2 * Vmu) * (x[0] - mu0) ** 2
            )

        lamc, g = proplam(h, sig2, rs=rs)
        MHprob = flam(lamc) - flam([mu, phi]) + g([mu, phi]) - g(lamc)
        if np.exp(MHprob) > rs.uniform():
            mu = lamc[0]
            phi = lamc[1]
            countlam += 1
        Hphi[Hphi_off_diag_idxs] = -phi

        if loop > burnin:
            i = loop - burnin - 1
            store_tau[i, :] = tau.T
            store_h[i, :] = h.T
            store_alp[i, :] = alp.T
            store_theta[i, :] = np.concatenate(
                [[mu, beta, phi, sig2], Omega[np.tril_indices_from(Omega)]]
            )
        if not loop % 10_000:
            print("Loop {}...".format(loop))

    print("MCMC takes {:.2f} seconds".format(time() - start_time))

    apt_rate = np.array([counth, countlam]) / nloop * 100
    print("Acceptance rate: h: {:.1f}%, (μ,φ): {:.1f}%".format(*apt_rate))

    return SVMDraws(
        theta=store_theta,  # [mu beta phi sig2 Omega([1 2 4])]
        tau=store_tau,
        alpha=store_alp,
        h=store_h,
        y=y,
        tid=tid,
        apt_rate=apt_rate,
    )

    # if is_dym_prob:
    #     compute_dyn_prob()
    #     plt.figure()
    #     plt.plot(tid, prob,'LineWidth',2,'Color','black')
    #     # box off;
    #     plt.ylim([0, 1.02])
    #     plt.xlim([np.min(tid)-1, 2014])
    #     plt.title(r'Dynamic Probabilities that $\alpha_t \neq 0$');


# This function obtains a proposal density for sampling (mu, phi) using a
# MH step
# Details: use the Newton-Raphason method (with BHHH) to maximize the
# conditional distribution of (mu, phi) parameterized as
# delta = (mu, tanh^(-1)(phi))
# outputs: a draw for (mu, phi) and the proposal density function
def proplam(h, sig2, rs):
    maxcount = 100
    T = h.shape[0]
    s = np.zeros([T, 2])
    mut = np.mean(h)
    phit = (h[:-1] - mut).T @ (h[1:] - mut) / np.sum((h[:-1] - mut) ** 2)
    delt = np.array([mut, np.arctanh(phit)]).T
    count = 0
    for count in range(maxcount):
        s[0, 0] = 1. / np.cosh(delt[1]) ** 2 / sig2 * (h[0] - delt[0])
        s[1:, 0] = (
            (1 - np.tanh(delt[1]))
            / sig2
            * (h[1:] - np.tanh(delt[1]) * h[:-1] - delt[0] * (1 - np.tanh(delt[1])))
        )
        s[0, 1] = np.tanh(delt[1]) / sig2 * (h[0] - delt[0]) / np.cosh(delt[1]) ** 2
        s[1:, 1] = (
            1
            / np.cosh(delt[1]) ** 2
            / sig2
            * (h[:-1] - delt[0])
            * (h[1:] - delt[0] - np.tanh(delt[1]) * (h[:-1] - delt[0]))
        )
        S = sum(s).T
        B = s.T @ s
        delt = delt + solve(B, S)
        count += 1
        # stopping criterion
        if np.sum(np.abs(solve(B, S))) < 1e-4:
            break
    C = np.linalg.cholesky(B)
    C_is_pd = np.all(np.linalg.eig(B)[0] > 0)  # inefficient, boo
    if not C_is_pd or count == maxcount - 1:
        print("not C_is_pd or maxcount reached")
        delt = np.array([mut, np.arctanh(phit)]).T
        s[0, 0] = 1. / np.cosh(delt[1]) ** 2 / sig2 * (h[0] - delt[0])
        s[1:, 0] = (
            (1 - np.tanh(delt[1]))
            / sig2
            * (h[1:] - np.tanh(delt[1]) * h[:-1] - delt[0] * (1 - np.tanh(delt[1])))
        )
        s[0, 1] = np.tanh(delt[1]) / sig2 * (h[0] - delt[0]) / np.cosh(delt[1]) ** 2
        s[1:, 1] = (
            1
            / np.cosh(delt[1]) ** 2
            / sig2
            * (h[:-1] - delt[0])
            * (h[1:] - delt[0] - np.tanh(delt[1]) * (h[:-1] - delt[0]))
        )
        B = s.T @ s
        C = np.linalg.cholesky(B)
    # t proposal with df nu = 5
    nu = 5
    δ = delt + solve(C.T, rs.normal(size=2)) / np.sqrt(
        rs.gamma(size=1, shape=nu / 2, scale=2 / nu)
    )
    lam = np.array([δ[0], np.tanh(δ[1])])
    c = (
        special.gammaln((nu + 2) / 2)
        - special.gammaln(nu / 2)
        - np.log(nu)
        - np.log(np.pi)
        + .5 * np.log(np.linalg.det(B))
    )

    def g(x):
        err = np.array([x[0], np.arctanh(x[1])]).T - delt
        return (
            c
            + 2 * np.log(np.cosh(np.arctanh(x[1])))  # Jacobian of transformation
            - (nu + 2.) / 2. * np.log(1. + 1. / nu * err.T @ B @ err)
        )

    return lam, g


class SVMDraws(dict):
    """Container for draws from running model."""

    def __getattr__(self, item):
        if isinstance(item, str) and item.endswith("hat"):
            return np.mean(self.get(item[:-3], None), axis=0)
        elif isinstance(item, str) and item.endswith("CI"):
            return np.percentile(a=self.get(item[:-2], None), q=[5., 95.], axis=0)
        else:
            return self.get(item, None)

    # https://stackoverflow.com/questions/2049849/why-cant-i-pickle-this-object
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def plot_data(self):
        plt.figure()
        plt.plot(self.tid, self.y)
        plt.title("$y_t$")
        plt.xlim((np.min(self.tid) - 1, 2014))

    def plot_states(self):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.box(False)
        plt.plot(self.tid, self.hhat, linewidth=2, color="black")
        for i in [0, 1]:
            plt.plot(self.tid, self.hCI[i, :], "--r", linewidth=2)
        plt.xlim((min(self.tid) - 1, 2014))
        plt.title("$h_t$")
        plt.subplot(1, 2, 2)
        plt.box(False)
        plt.plot(self.tid, self.alphahat, linewidth=2, color="black")
        for i in [0, 1]:
            plt.plot(self.tid, self.alphaCI[i, :], "--r", linewidth=2)
        plt.xlim((min(self.tid) - 1, 2014))
        plt.title(r"$\alpha_t$")

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    @staticmethod
    def load(filename):
        if "__file__" in globals():
            base = os.path.dirname(__file__)
        else:
            base = os.path.join("ptvi")
        internal_file = os.path.join(base, "data", filename)
        if os.path.isfile(filename):
            return pickle.load(open(filename, "rb"))
        elif os.path.isfile(internal_file):
            return SVMDraws.load(internal_file)
        else:
            raise Exception("File {} not found.".format(filename))

    def summary(self):
        _thetaCI = self.thetaCI
        summary = pd.DataFrame(
            {"Mean": self.thetahat, "5%": _thetaCI[0, :], "95%": _thetaCI[1, :]},
            index=["μ", "β", "φ", "σ²", "Ω_α", "Ω_ατ", "Ω_τ"],
        )
        return summary

    def display(self):
        return self.summary().display()

    def __repr__(self):
        return "***** SVM Model MCMC Summary *****\n\n{}".format(repr(self.summary()))


def vec_from_clipboard():
    return np.array(pd.read_clipboard(header=None)).squeeze()


def load_cpi():
    if "__file__" in globals():
        base = os.path.dirname(__file__)
    else:
        base = os.path.join("ptvi")
    cpi_file = os.path.join(base, "data", "USCPI.csv")
    cpi = pd.read_csv(cpi_file, names=["USCPI"], header=None)
    cpi.index = pd.date_range("1948Q1", end="2013Q3", freq="Q")
    return cpi


def main():
    cpi = load_cpi()
    y = cpi["USCPI"]
    rs = np.random.RandomState(seed=123)
    samples = svm_sampler(y, nloop=55_000, burnin=10_000, rs=rs)
    samples.save("USCPI_samples.p")
    # load with:
    # samples = SVMDraws.load('USCPI_samples.p')


# if __name__ == '__main__':
#     main()
