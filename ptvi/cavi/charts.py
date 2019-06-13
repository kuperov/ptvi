import matplotlib.pyplot as plt
import numpy as np


def fan_chart(
    y,
    fc,
    ax=None,
    probs=None,
    start=None,
    x=None,
    origin_line=True,
    legend=True,
    **kwargs,
):
    """Plot a symmetric fan chart from a data series and forecast sequence.

    Args:
        y:      data series
        fc:     sequence of forecasts, as dict like {t: draw array}
        ax:     axis to plot on
        start:  number of observation to start at (default: first)
        probs:  quantiles of distribution to include in fan chart, where
                0 = median line and 0.9 = central 90 per cent. Default
                is [.9, .5, 0]
        origin_line: if true, draw a black vertical line at the forecast origin.
        legend: if true, render a chart legend

    Returns:
        Matplotlib chart object, if no axis given, or else None
    """
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, **kwargs)
    N, steps = len(y), len(fc)
    assert all(s in fc for s in range(N, N + steps))
    if x:
        assert len(x) == N + steps
    else:
        x = np.array(range(1, N + steps + 1))
    if not start:
        start = np.min(x)
    ax.set_xlim([start, np.max(x)])
    ax.plot(x[:N], y, color="black", alpha=1, linewidth=2)
    if not probs:
        probs = [0.9, 0.5, 0]
    if 0 in probs:
        probs = probs.copy()  # don't modify caller's list
        probs.remove(0)
        median_line = [np.median(fc[x]) for x in range(N, N + steps)]
        ax.plot(x[N:], median_line, color="black", alpha=1, linewidth=2, label="Median")
    fan_xs = np.array(range(N, N + steps + 1))
    quantiles = sorted([q for p in probs for q in [0.5 * (1 - p), 0.5 * (1 + p)]])
    prob_list = probs + list(reversed(list(probs[:-1])))
    segments = zip(quantiles[:-1], quantiles[1:], prob_list)
    for bottom_prob, top_prob, segment_prob in segments:
        alpha = 1 - segment_prob
        label = f"{100*segment_prob:.0f}%" if top_prob > 0.5 else None
        upper_y = np.r_[
            y[-1], [np.quantile(fc[x], q=top_prob) for x in range(N, N + steps)]
        ]
        lower_y = np.r_[
            y[-1], [np.quantile(fc[x], q=bottom_prob) for x in range(N, N + steps)]
        ]
        ax.fill_between(
            fan_xs, upper_y, lower_y, color="blue", alpha=alpha, label=label
        )
    if legend:
        ax.legend()
    if origin_line:
        ax.axvline(x=N, color="black", linestyle=":", linewidth=1)
    return fig


def discrete_fan_chart(
    y,
    fc,
    ax=None,
    probs=None,
    start=None,
    x=None,
    origin_line=True,
    legend=True,
    **kwargs,
):
    """Plot a symmetric fan chart from a data series and discrete forecast sequence.

    The forecasts are presented as a histogram where the cells of the 1D array are
    zero-based bins.

    Args:
        y:      data series
        fc:     sequence of forecasts, as dict like {t: histogram}
        ax:     axis to plot on
        start:  number of observation to start at (default: first)
        probs:  quantiles of distribution to include in fan chart, where
                0 = median line and 0.9 = central 90 per cent. Default
                is [.9, .5, 0]
        origin_line: if true, draw a black vertical line at the forecast origin.
        legend: if true, render a chart legend

    Returns:
        Matplotlib chart object, if no axis given, or else None
    """
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, **kwargs)
    N, steps = len(y), len(fc)
    assert all(s in fc for s in range(N, N + steps))
    if x:
        assert len(x) == N + steps
    else:
        x = np.array(range(1, N + steps + 1))
    if not start:
        start = np.min(x)
    start_idx = np.where(x == start)[0][0]
    ax.set_xlim([start, np.max(x)])
    ax.plot(x[start_idx:N], y[start_idx:], color="black", alpha=1, linewidth=2)
    if not probs:
        probs = [0.9, 0.5, 0]
    fan_xs = np.array(range(N, N + steps + 1))
    quantiles = sorted([q for p in probs for q in [(1 - p) / 2, 1 - (1 - p) / 2]])
    prob_list = probs + list(reversed(list(probs[:-1])))
    segments = zip(quantiles[:-1], quantiles[1:], prob_list)
    segments = []
    for i in range(len(prob_list)):
        segments.append((quantiles[i], quantiles[i + 1], prob_list[i]))
    for bottom_prob, top_prob, segment_prob in segments:
        alpha = 1 - segment_prob
        if top_prob > 0.5:
            label = f"{100*segment_prob:.0f}%"
        elif segment_prob == 0.0:
            label = "Median"
        else:
            label = None
        upper_y = np.r_[
            y[-1],
            [
                discrete_hist_qtile(fc[x], q=top_prob, mode="ceiling")
                for x in range(N, N + steps)
            ],
        ]
        lower_y = np.r_[
            y[-1],
            [
                discrete_hist_qtile(fc[x], q=bottom_prob, mode="floor")
                for x in range(N, N + steps)
            ],
        ]
        ax.fill_between(
            fan_xs, upper_y, lower_y, color="blue", alpha=alpha, label=label
        )
    if legend:
        ax.legend()
    if origin_line:
        ax.axvline(x=N, color="black", linestyle=":", linewidth=1)
    return fig


def fan_chart_from_draws(
    y,
    fc,
    ax=None,
    probs=None,
    start=None,
    x=None,
    origin_line=True,
    legend=True,
    **kwargs,
):
    """Plot a symmetric fan chart from a data series and forecast sequence.

    Args:
        y:      data series
        fc:     numpy 2-array of forecast paths: periods in dimension 0 and 
                draws in dimension 1
        ax:     axis to plot on
        start:  number of observation to start at (default: first)
        probs:  quantiles of distribution to include in fan chart, where
                0 = median line and 0.9 = central 90 per cent. Default
                is [.9, .5, 0]
        origin_line: if true, draw a black vertical line at the forecast origin.
        legend: if true, render a chart legend

    Returns:
        Matplotlib chart object, if no axis given, or else None
    """
    fig = None
    if not ax:
        fig, ax = plt.subplots(1, 1, **kwargs)
    N, steps = len(y), fc.shape[0]
    if x:
        assert len(x) == N + steps
    else:
        x = np.array(range(1, N + steps + 1))
    if not start:
        start = np.min(x)
    ax.set_xlim([start, np.max(x)])
    ax.plot(x[:N], y, color="black", alpha=1, linewidth=2)
    if not probs:
        probs = [0.9, 0.5, 0]
    if 0 in probs:
        probs = probs.copy()  # don't modify caller's list
        probs.remove(0)
        median_line = [y[-1]] + [np.median(fc[i, :]) for i in range(steps)]
        ax.plot(x[(N-1):], median_line, color="black", alpha=1, linewidth=2, label="Median")
    fan_xs = np.array(range(N, N + steps + 1))
    quantiles = sorted([q for p in probs for q in [0.5 * (1 - p), 0.5 * (1 + p)]])
    prob_list = probs + list(reversed(list(probs[:-1])))
    segments = list(zip(quantiles[:-1], quantiles[1:], prob_list))
    for bottom_prob, top_prob, segment_prob in segments:
        alpha = 1 - segment_prob
        label = f"{100*segment_prob:.0f}%" if top_prob > 0.5 else None
        upper_y = np.r_[
            y[-1], [np.percentile(fc[i, :], q=100*top_prob) for i in range(steps)]
        ]
        lower_y = np.r_[
            y[-1], [np.percentile(fc[i, :], q=100*bottom_prob) for i in range(steps)]
        ]
        ax.fill_between(
            fan_xs, upper_y, lower_y, color="blue", alpha=alpha, label=label
        )
    if legend:
        ax.legend()
    if origin_line:
        ax.axvline(x=N, color="black", linestyle=":", linewidth=1)
    return fig


def _hist_qtile(hist, q):
    """Compute quantile for a discrete histogram, interpolating between buckets.

    Args:
        hist:    1-array of weights that sum to 1. First bucket is zero.
        q:       desired quantile, 0.0 <= q <= 1.0
    
    Returns:
        Quantile (fractional bucket value) corresponding to q
    """
    assert 0.0 <= q <= 1.0
    bins = len(hist)
    xs = list(range(bins + 1))
    ys = np.r_[0, np.cumsum(hist)]
    for x1 in range(bins):
        if ys[x1] <= q <= ys[x1 + 1]:
            return x1 + (q - ys[x1]) / (ys[x1 + 1] - ys[x1])
    return ys[-1]


# Vectorize quantile function on the desired quantiles
hist_qtile = np.vectorize(_hist_qtile, otypes=[np.float64], excluded=[0])


def discrete_hist_qtile(hist, q, mode="floor"):
    """Compute discrete quantile for a discrete histogram.

    Args:
        hist:    1-array of weights that sum to 1. First bucket is zero.
        q:       desired quantile, 0.0 <= q <= 1.0
        mode:    if 'floor' return largest bucket smaller than q, vice-versa for 'ceiling'
    
    Returns:
        Quantile (fractional bucket value) corresponding to q
    """
    assert 0.0 <= q <= 1.0
    if mode == "floor":
        return np.floor(hist_qtile(hist, q))
    elif mode == "ceiling":
        return np.ceil(hist_qtile(hist, q))
    else:
        raise Exception(f'Unknown mode "{mode}".')


def trace(arr, name="", title=None, **kwargs):
    """MCMC trace plot for the given array.

    Args:
        arr:   ndarray of draws. Dimension 0 is MCMC iteration.
        name:  string or list of strings describing series
        title: plot title
    """
    assert arr.ndim > 0
    if arr.ndim == 1:  # univariate variable
        arr = np.expand_dims(arr, 1)
        name = [name]

    fig, axes = plt.subplots(nrows=arr.shape[1], **kwargs)
    for i, ax in enumerate(axes):
        name = name[i] if type(name) == "list" else name
        ax.plot(arr[:, i], label=name)
        if name:
            plt.legend()
    if title:
        plt.title(title)
    plt.show()
