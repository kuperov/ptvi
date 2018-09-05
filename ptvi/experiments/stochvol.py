import random
import json
import os
from datetime import datetime
from time import time
import matplotlib

matplotlib.use("Agg")  # must be before other graphics imports
import matplotlib.pyplot as plt
import click
import pandas as pd
import torch
from torch.distributions import Normal
from scipy import stats
import numpy as np
from ptvi import (
    FilteredStateSpaceModel,
    global_param,
    AR1Proposal,
    PFProposal,
    LogNormalPrior,
    NormalPrior,
    BetaPrior,
    ModifiedBetaPrior,
    sgvb,
)


_DIVIDER = "―" * 80


class SVModel(FilteredStateSpaceModel):
    """ A simple stochastic volatility model for estimating with FIVO.

    .. math::
        x_t = exp(a)exp(z_t/2) ε_t       ε_t ~ Ν(0,1)
        z_t = b + c * z_{t-1} + ν_t    ν_t ~ Ν(0,1)

    The proposal density is also an AR(1):

    .. math::
        z_t = d + e * z_{t-1} + η_t    η_t ~ Ν(0,1)
    """

    class SVModelResult(FilteredStateSpaceModel.FIVOResult):
        def forecast(self, steps=1, n=100):
            """Produce forecasts of the specified number of steps ahead.

            Procedure: sample ζ, filter to get p(z_T | y, θ), project the state chain
            forward, then compute y.
            """
            z_T_draws = torch.zeros(n)
            z_proj_draws = torch.zeros((n, steps))
            y_proj_draws = torch.zeros((n, steps))
            sample = torch.zeros((1, n))
            self.model.num_particles = n  # dodgy hack to simulate more particles
            phatns = torch.zeros((n,))
            for i in range(n):
                ζ = self.q.sample()
                a, b, c = ζ[0], ζ[1], ζ[2]
                phatns[i] = self.model.simulate_log_phatN(self.y, ζ, sample)
                # just take a single particle's z_T
                z_T_draws[i] = sample[0, random.randint(0, n - 1)]
                z_proj_draws[i, 0] = b + c * z_T_draws[i] + torch.randn(1)
                for j in range(1, steps):
                    z_proj_draws[i, j] = b + c * z_proj_draws[i, j - 1] + torch.randn(1)
                y_proj_draws[i, :] = Normal(
                    0, torch.exp(a) * torch.exp(z_proj_draws[i, :] / 2)
                ).sample()

            kde = stats.gaussian_kde(y_proj_draws[:, -1].cpu().numpy())
            return kde, y_proj_draws.cpu().numpy()

    name = "Particle filtered stochastic volatility model"
    a = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="α")
    b = global_param(prior=NormalPrior(0, 1))
    c = global_param(prior=ModifiedBetaPrior(0.5, 1.5), transform="logit", rename="ψ")
    d = global_param(prior=NormalPrior(0, 1))
    e = global_param(prior=BetaPrior(1, 1), transform="logit", rename="ρ")
    f = global_param(prior=LogNormalPrior(0, 1), transform="log", rename="ι")

    result_type = SVModelResult

    def simulate(self, a, b, c):
        """Simulate from p(x, z | θ)"""
        a, b, c = map(torch.tensor, (a, b, c))
        z = torch.empty((self.input_length,))
        z[0] = Normal(b, (1 - c ** 2) ** (-.5)).sample()
        for t in range(1, self.input_length):
            z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        x = Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()
        return x.type(self.dtype).to(self.device), z.type(self.dtype).to(self.device)

    def conditional_log_prob(self, t, y, z, ζ):
        """Compute log p(x_t, z_t | y_{0:t-1}, z_{0:t-1}, ζ).

        Args:
            t: time index (zero-based)
            y: y_{0:t} vector of points observed up to this point (which may
               actually be longer, but should only be indexed up to t)
            z: z_{0:t} vector of unobserved variables to condition on (ditto,
               array may be longer)
            ζ: parameter to condition on; should be unpacked with self.unpack
        """
        a, b, c, _, _, _ = self.unpack_natural(ζ)
        if t == 0:
            log_pzt = Normal(b, (1 - c ** 2) ** (-.5)).log_prob(z[t])
        else:
            log_pzt = Normal(b + c * z[t - 1], 1).log_prob(z[t])
        log_pxt = Normal(0, torch.exp(a) * torch.exp(z[t] / 2)).log_prob(y[t])
        return log_pzt + log_pxt

    def sample_observed(self, ζ, y, fc_steps=0):
        a, _, _, _, _, _ = self.unpack_natural(ζ)
        z = self.sample_unobserved(ζ, y, fc_steps)
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def sample_unobserved(self, ζ, y, fc_steps=0):
        assert y is not None
        a, b, c, _, _, _ = self.unpack_natural(ζ)
        # get a sample of states by filtering wrt y
        z = torch.empty((len(y) + fc_steps,))
        self.simulate_log_phatN(y=y, ζ=ζ, sample=z)
        # now project states forward fc_steps
        if fc_steps > 0:
            for t in range(self.input_length, self.input_length + fc_steps):
                z[t] = b + c * z[t - 1] + Normal(0, 1).sample()
        return Normal(0, torch.exp(a) * torch.exp(z / 2)).sample()

    def proposal_for(self, y: torch.Tensor, ζ: torch.Tensor) -> PFProposal:
        _, _, _, d, e, f = self.unpack_natural(ζ)
        return AR1Proposal(μ=d, ρ=e, σ=f)

    def __repr__(self):
        return (
            f"Stochastic volatility model with parameters {{a, b, c}}:\n"
            f"\ty_t = exp(a * z_t/2) ε_t        t=1,…,{self.input_length}\n"
            f"\tz_t = b + c * z_{{t-1}} + ν_t,    t=2,…,{self.input_length}\n"
            f"\tz_1 = b + 1/√(1 - c^2) ν_1\n"
            f"\twhere ε_t, ν_t ~ Ν(0,1)\n\n"
            f"Filter with {self.num_particles} particles; AR(1) proposal params {{d, e, f}}:\n"
            f"\tz_t = d + e * z_{{t-1}} + f η_t,  t=2,…,{self.input_length}\n"
            f"\tz_1 = d + f/√(1 - e^2) η_1\n"
            f"\twhere η_t ~ Ν(0,1)\n"
        )


@click.group()
@click.option("--algo_seed", default=1, help="Seed for algorithm")
@click.option("--data_seed", default=1, help="Seed for data generation")
@click.pass_context
def stochvol(ctx, algo_seed, data_seed):
    """Utility for running simulations with the stochastic volatility model."""
    ctx.obj["algo_seed"] = algo_seed
    ctx.obj["data_seed"] = data_seed


@stochvol.command()
@click.argument("filename")
@click.argument("t", type=click.INT)
@click.argument("reps", type=click.INT)
@click.option("--a", default=1., help="True a parameter value")
@click.option("--b", default=0., help="True b parameter value")
@click.option("--c", default=.95, help="True c parameter value")
@click.pass_context
def multiple(ctx, filename, t, reps, a, b, c):
    """Generate REPS data sets of length T and store in <filename> in CSV format.

    The data seed for each will be {seed+0, seed+1, ..., seed+T-1}. Each dataset gets
    one column, and the columns are named by their respective ordinal numbers,
    {"1", "2", ..., "<REPS>"}.

    All datasets are generated from the same true values, {a, b, c}.

    The FILENAME argument specifies the CSV file to write to. T is the number of
    observations to generate. REPS is the number of series to generate.

    Example:

        stochvol multiple experiment.csv 1000 500 --a=0.5 --b=0.25 --c=0.8

    """
    seed = ctx.obj["data_seed"]
    click.echo(
        f"Generating {reps} series of length {t} with seed = {{{seed}, {seed+1}, ...}}"
    )
    click.echo(f"True values: a={a}, b={b}, c={c}.")
    params = dict(a=a, b=b, c=c)
    model = SVModel(input_length=t, **params)
    click.echo(repr(model))
    dataset = {"t": list(range(1, t + 1))}
    for r in range(reps):
        y, xs = model.simulate(**params)
        dataset[str(r + 1)] = y
    df = pd.DataFrame(data=dataset, index=list(range(1, t + 1)))
    df.to_csv(filename, index=False)


@stochvol.command()
@click.argument("filename")
@click.argument("t", type=click.INT)
@click.option("--a", default=1., help="True a parameter value")
@click.option("--b", default=0., help="True b parameter value")
@click.option("--c", default=.95, help="True c parameter value")
@click.pass_context
def generate(ctx, filename, t, a, b, c):
    """Generate one data set of length T and store in <filename> in CSV format.

    The FILENAME argument specifies the CSV file to write to. T is the number of
    observations to generate.

    The data seed for each will be <seed>. The columns are 'y' for the observed data
    and 'z' for the latent series. The true values are {a, b, c}.

    Example:

        stochvol --data_seed=123 generate experiment.csv 10000 --a=0.5 --b=0.25 --c=0.8

    """
    seed = ctx.obj["data_seed"]
    click.echo(_DIVIDER)
    click.echo(f"Generating series of length {t} with seed = {seed}.")
    click.echo(f"True values: a={a}, b={b}, c={c}.")
    params = dict(a=a, b=b, c=c)
    model = SVModel(input_length=t)
    click.echo(repr(model))
    y, z = model.simulate(**params)
    dataset = {"t": list(range(1, t + 1)), "y": y, "z": z}
    df = pd.DataFrame(data=dataset, index=dataset["t"])
    df.to_csv(filename, index=False)
    click.echo("Done.")
    click.echo(_DIVIDER)


@stochvol.command()
@click.argument("datafile")
@click.argument("t", type=click.INT)
@click.argument("outfile")
@click.option("--N", default=100, help="Reps for estimating score")
@click.option("--a", default=1., help="True a parameter value")
@click.option("--b", default=0., help="True b parameter value")
@click.option("--c", default=.95, help="True c parameter value")
@click.option("--maxiters", default=1_000_000, help="Maximum iterations")
@click.pass_context
def conditional(ctx, datafile, t, outfile, n, a, b, c, maxiters):
    """Forecast stoch vol model and compute log score, conditional on T observations.

    Example:

        stochvol --data_seed=123 --algo_seed=123 conditional experiment.csv 200 SV00200.json --N=100 --a=1. --b=0. --c=0.8
    """
    assert t > 1
    start_date, start_time = str(datetime.today()), time()
    click.echo(_DIVIDER)
    click.echo("Stochastic volatility model: conditional score estimation")
    click.echo(_DIVIDER)
    true_params = dict(a=a, b=b, c=c)
    algo_seed = ctx.obj["algo_seed"]
    data_seed = ctx.obj["data_seed"]
    data = pd.read_csv(datafile)
    click.echo(f"Started at: {start_date}")
    click.echo(f"Reading {t}/{len(data)} observations from {datafile}.")
    click.echo(f"True parameters assumed to be a={a}, b={b}, c={c}")

    # draw N variates from p(y_T+1 | z_T+1, a, b, c)
    click.echo(
        f"Drawing {n} variates from p(y_T+1, z_T+1 | z_T, a, b, c) with "
        f"data_seed={data_seed}"
    )
    torch.manual_seed(data_seed)
    a, b, c = map(torch.tensor, (a, b, c))
    z_next = b + c * data["z"][t - 1] + Normal(0, 1).sample((n,))
    y_next = Normal(0, torch.exp(a) * torch.exp(z_next / 2)).sample()
    y_next_list = y_next.cpu().numpy().squeeze().tolist()  # for saving

    # perform inference
    y = data["y"][:t]
    model = SVModel(input_length=t)
    click.echo(repr(model))
    torch.manual_seed(algo_seed)
    fit = sgvb(model, y, max_iters=maxiters)
    click.echo("Inference summary:")
    click.echo(fit.summary(true=true_params))

    click.echo(f"Generating {n} forecast draws from q...")
    # filter to get p(z_T | y, θ) then project z_{T+1}, z_{T+2}, ...
    forecast, fc_draws = fit.forecast(steps=1)
    fc_draws_list = fc_draws.squeeze().tolist()

    dens = forecast.pdf(y_next)
    scores = np.log(dens[dens > 0])
    score = np.mean(scores)
    score_se = np.std(scores)
    click.echo(f"Forecast log score = {score:.4f} nats (sd = {score_se:.4f}, n = {n})")

    click.echo(f"Writing results to {outfile} in JSON format.")
    y_list = data["y"][:t].tolist()
    z_list = data["z"][:t].tolist()
    summary = {
        "method": "VSMC",
        "algo_seed": algo_seed,
        "data_seed": data_seed,
        "datafile": datafile,
        "t": t,
        "outfile": outfile,
        "fc_draws": fc_draws_list,
        "score": score,
        "score_se": score_se,
        "n": n,
        "y_next": y_next_list,
        "start_date": start_date,
        "elapsed": time() - start_time,
        "true_params": true_params,
        "full_length": len(data),
        "max_iters": maxiters,
        "inference_results": str(fit.summary()),
        "y": y_list,
        "z": z_list,
    }
    with open(outfile, "w", encoding="utf8") as ofilep:
        json.dump(summary, ofilep, indent=4, sort_keys=True)

    click.echo(f"Done in {time() - start_time:.1f} seconds.")
    click.echo(_DIVIDER)


@stochvol.command()
@click.argument("jsonfile")
@click.argument("outfile")
@click.pass_context
def mcmc(ctx, jsonfile, outfile):
    """Forecasts SV model using MCMC procedure.

    The JSONFILE parameter is the file produced by 'conditional'. Results are saved
    to OUTFILE.

    Example:

        stochvol mcmc experiment.json results.json

    """
    import rpy2.robjects as robjects

    start_date, start_time = str(datetime.today()), time()
    scriptfile = os.path.join(os.path.dirname(__file__), "stochvol.R")
    with open(scriptfile, "r") as content_file:
        script = content_file.read()
        robjects.r(script)
    click.echo(_DIVIDER)
    click.echo(f"MCMC stochastic volatility model")
    click.echo(f"Started at: {start_date}")
    click.echo(f"Reading {jsonfile} and writing to {outfile}")
    mcmc = robjects.globalenv["mcmc_SV"]
    mcmc(jsonfile, outfile)
    click.echo(f"Done in {time() - start_time:.1f} seconds.")
    click.echo(_DIVIDER)


@stochvol.command()
@click.argument("jsonfiles", nargs=-1)
@click.argument("pdffile", nargs=1)
@click.pass_context
def compare(ctx, jsonfiles, pdffile):
    results = []
    for file in jsonfiles:
        with open(file, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    fcs = [stats.gaussian_kde(r["fc_draws"]) for r in results]
    # simulate to figure out range of plot
    rord = np.concatenate([fc.resample(50) for fc in fcs])
    range = (rord.min(), rord.max())
    # plot densities
    xs = np.linspace(*range, num=1000)
    for fc, res in zip(fcs, results):
        lbl = f'{res["method"]} (log score = {res["score"]:.1f})'
        plt.plot(xs, fc.pdf(xs), label=lbl)
    plt.title(f'Forecasts: T={results[0]["t"]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdffile)


def main():
    stochvol(obj={})


if __name__ == "__main__":
    main()
