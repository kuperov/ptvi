from typing import List
import numpy as np


class StoppingHeuristic(object):
    """Abstract class for early stopping heuristic.
    """

    def early_stop(self, est_elbo: float) -> bool:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class NullStoppingHeuristic(StoppingHeuristic):
    """Null heuristic that never stops inference early."""

    def early_stop(self, est_elbo: float):
        return False

    def __str__(self):
        return "Null stopping heuristic (never fires)"


class ExponentialStoppingHeuristic(StoppingHeuristic):
    """Implements trailing average early stopping heuristic.

    Every N iterations, computes an exponentially-weighted average of the
    estimated elbo. If this has not increased in N*M iterations, early_stop()
    returns true. A 'true' return can't happen before 2*N*M iterations.
    """

    def __init__(self, N: int, M: int, α: float = 0.05):
        assert 0. < α <= 1.
        self.N, self.M, self.α = N, M, α
        self.i = -1
        self.circular_buffer: List[float] = [None] * M
        self.curr_elbo, self.past_elbo = None, None

    def early_stop(self, est_elbo: float) -> bool:
        self.i += 1
        if self.i % self.N:
            return False  # only check every N iterations
        if self.curr_elbo is not None:
            self.curr_elbo = est_elbo * self.α + (1. - self.α) * self.curr_elbo
        else:
            self.curr_elbo = est_elbo
        buf_idx = (self.i // self.N) % self.M
        # TODO: consider the variance of this measure too
        if self.i > self.N * self.M:
            assert self.circular_buffer[buf_idx] is not None
            if self.circular_buffer[buf_idx] > self.curr_elbo:
                return True  # stahp!
        self.circular_buffer[buf_idx] = self.curr_elbo

    def __str__(self):
        return (
            f"Exponential stopping heuristic (N={self.N}, M={self.M}, " f"α={self.α})"
        )


class NoImprovementStoppingHeuristic(StoppingHeuristic):
    """Stop if there has been no improvement for <patience> steps.

    Every N iterations, computes an exponentially-weighted average of the
    estimated elbo. If this has not increased in the last <patience> steps,
    early_stop() returns true.
    """

    def __init__(self, patience=10, skip=1, min_steps=100, ε=.01, α=.1):
        assert 0. < α <= 1.
        self.skip, self.patience, self.α, self.ε = skip, patience, α, ε
        self.min_steps = min_steps
        self.i = -1
        self.no_improvement_count = 0
        self.curr_elbo, self.past_elbo = None, None

    def early_stop(self, est_elbo: float) -> bool:
        self.i += 1
        if self.i % self.skip:
            return False  # only check every N iterations
        if self.curr_elbo is not None:
            self.curr_elbo = est_elbo * self.α + (1. - self.α) * self.curr_elbo
            if est_elbo > self.curr_elbo + self.ε:
                self.no_improvement_count = 0
            elif self.i > self.min_steps:
                self.no_improvement_count += 1
        else:
            self.curr_elbo = est_elbo
        return self.no_improvement_count == self.patience

    def __str__(self):
        return (
            f"Stop on no improvement (skip={self.skip}, "
            f"patience={self.patience}, min_steps={self.min_steps}, "
            f"ε={self.ε}, α={self.α})"
        )


class MedianGrowthStoppingHeuristic(StoppingHeuristic):
    """Impose a minimum rate of improvement in the median elbo.

    Every <skip> steps, we compute the median elbo estimate over the past
    <skip>*<window> elbo evaluations. If this median has not increased by at
    least ε in the last skip*patience steps, early_stop() returns true.
    """

    def __init__(
        self, patience: int = 10, skip: int = 1, min_steps: int = 100, ε: float = .1
    ):
        assert min_steps > 2 * patience
        self.skip, self.patience, self.ε = skip, patience, ε
        self.min_steps = min_steps
        self.i = -1
        self.elbo_circular_buffer: List[float] = [None] * patience
        self.median_elbo_circular_buffer: List[float] = [None] * patience
        self.no_improvement_count = 0

    def early_stop(self, curr_elbo: float) -> bool:
        self.i += 1
        if self.i % self.skip:
            return False  # only check every N iterations
        buf_idx: int = (self.i // self.skip) % self.patience
        self.elbo_circular_buffer[buf_idx] = curr_elbo
        if self.i <= self.patience:
            return False
        this_median = np.median(self.elbo_circular_buffer)
        if (
            self.i > self.patience * 2
            and self.median_elbo_circular_buffer[buf_idx] + self.ε > this_median
        ):
            self.no_improvement_count = 0
        elif self.i > self.patience * 2:
            self.no_improvement_count += 1
        self.median_elbo_circular_buffer[buf_idx] = this_median
        return self.no_improvement_count == self.patience and self.i > self.min_steps

    def __str__(self):
        return (
            f"Minimum median elbo improvement rate (min_steps={self.min_steps},"
            f" patience={self.patience}, skip={self.skip}, ε={self.ε})"
        )


class SupGrowthStoppingHeuristic(StoppingHeuristic):
    """Impose a minimum rate of improvement in the elbo supremum.

    If the best has not increased by a rate of at least ε in the last
    skip*patience steps, early_stop() returns true.
    """

    def __init__(
        self,
        patience: int = 20,
        skip: int = 10,
        min_steps: int = 50,
        ε: float = 1e-1,
        α: float = .1,
    ):
        assert min_steps > patience
        self.skip, self.patience, self.ε, self.α = skip, patience, ε, α
        self.min_steps = min_steps
        self.i = -1
        self.sup_elbo, self.curr_elbo = None, None
        self.best_circ_buf: List[float] = [None] * patience
        self.no_improvement_count = 0

    def early_stop(self, noisy_elbo: float) -> bool:
        self.i += 1
        if self.i == self.min_steps * self.skip:
            self.sup_elbo = self.curr_elbo  # in case we started in ok position
            self.no_improvement_count = 0
            self.best_circ_buf = [self.curr_elbo] * self.patience
        if self.i % self.skip:
            return False  # only check every N iterations
        if self.curr_elbo is None:
            self.curr_elbo = float(noisy_elbo)
        else:
            self.curr_elbo = self.α * float(noisy_elbo) + (1 - self.α) * self.curr_elbo
        buf_idx: int = (self.i // self.skip) % self.patience
        if self.sup_elbo is None or self.curr_elbo > self.sup_elbo:
            self.sup_elbo = self.curr_elbo
        if (
            self.i > self.patience * self.skip
            and self.sup_elbo - self.best_circ_buf[buf_idx] > self.ε
        ):
            self.no_improvement_count = 0
        elif self.i > self.patience * self.skip:
            self.no_improvement_count += 1
        self.best_circ_buf[buf_idx] = self.sup_elbo
        return (
            self.no_improvement_count >= self.patience
            and self.i > self.min_steps * self.skip
        )

    def __str__(self):
        return (
            f"Minimum supremum growth heuristic ("
            f"min_steps={self.min_steps*self.skip},"
            f" patience={self.patience}, skip={self.skip}, ε={self.ε})"
        )
