import torch
import math
import prettytable as pt


def print_results(values, true_values=None):
    headers = ["Variable", "Estimate"]
    if true_values:
        headers += ["True value"]
    values = values.copy()
    tbl = pt.PrettyTable(headers)

    def rnd(x):
        if x is None:
            return None
        else:
            return round(float(x), 5)

    for name, value in values.items():
        if name.startswith("log_"):
            name = name[4:]
            value = torch.exp(value)
        if name.endswith("_rev"):
            name = name[:-4]
            value = torch.tensor(list(reversed(value.tolist())))
        trueval = true_values.get(name)
        if trueval is not None:
            if type(trueval) == float or len(trueval) <= 1:
                trueval = [trueval]
            else:
                trueval = trueval.tolist()
        else:
            trueval = [None] * len(value)
        n = len(value)
        for i, val, true in zip(range(n), value.tolist(), trueval):
            if n > 1:
                tbl.add_row([f"{name}{i}", rnd(val), rnd(true)])
            else:
                tbl.add_row([name, rnd(val), rnd(true)])
    print(tbl)


def summary_row(values, iteration=None):
    """Print a one-row summary of values.

    Useful for writing progress during optimization.
    """
    myval = values.copy()
    elems = []
    if iteration is not None:
        elems.append(f"{iteration:4d}.")
    if "loss" in myval:
        elems.append(f'loss = {values["loss"]:4.4f}')
        del myval["loss"]
    elif "llik" in myval:
        elems.append(f'log lhood = {values["llik"]:4.4f}')
        del myval["llik"]
    elif "nllik" in myval:
        elems.append(f'log lhood = {-values["nllik"]:4.4f}')
        del myval["nllik"]
    for k, v in myval.items():
        elems.append(value_tuple(k, v))
    return "  ".join(elems)


def value_tuple(name, value):
    """Convert (name, value) into name: value string.

    If name begins with log_ then exponentiate first.
    """
    if name.endswith("_rev"):
        name = name[:-4]
        value = torch.tensor(list(reversed(value.tolist())))
    if name.startswith("log_"):
        name = name[4:]
        fmt = lambda x: f"{math.exp(float(x)):.2f}"
    else:
        fmt = lambda x: f"{float(x):.2f}"
    if type(value) == torch.Tensor:
        if value.ndimension() == 0:
            return f"{name}: {fmt(value)}"
        elif value.ndimension() == 1 and len(value) == 1:
            return f"{name}: {fmt(value[0])}"
        elif value.ndimension() == 1:
            vec = ", ".join([fmt(x) for x in value])
            return f"{name}: [{vec}]"
        else:
            return f"{name}: {fmt(value)}"
    else:
        return f"{name}: {fmt(value)}"
