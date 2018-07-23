from typing import List


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
        return 'Null stopping heuristic (never fires)'


class ExponentialStoppingHeuristic(StoppingHeuristic):
    """Implements trailing average early stopping heuristic.

    Every N iterations, computes an exponentially-weighted average of the
    estimated elbo. If this has not increased in N*M iterations, early_stop()
    returns true. A 'true' return can't happen before 2*N*M iterations.
    """

    def __init__(self, N: int, M: int, α: float=0.05):
        assert 0. < α <= 1.
        self.N, self.M, self.α = N, M, α
        self.i = -1
        self.circular_buffer: List[float] = [None] * M
        self.curr_elbo, self.past_elbo = None, None

    def early_stop(self, est_elbo: float) -> bool:
        self.i += 1
        if self.i % self.N: return False  # only check every N iterations
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
        return (f'Exponential stopping heuristic (N={self.N}, M={self.M}, '
                f'α={self.α})')


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
        if self.i % self.skip: return False  # only check every N iterations
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
            f'Stop on no improvement (skip={self.skip}, patience={self.patience},'
            f' min_steps={self.min_steps}, ε={self.ε}, α={self.α})')
