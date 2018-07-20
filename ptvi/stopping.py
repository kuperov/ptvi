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
