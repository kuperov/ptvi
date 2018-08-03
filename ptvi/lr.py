import torch


def make_advi_step_sequence(α=0.1, η=1., τ=1., ε=1e-16):
    """Step size sequence as desribed in:
    Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017).
    Automatic Differentiation Variational Inference. Journal of Machine
    Learning Research, 18, 1–45. https://doi.org/10.3847/0004-637X/819/1/50
    """
    state = {"i": 0}

    def ρ_generator(g):
        i = state["i"] + 1
        if "s" not in state:
            s = g ** 2
        else:
            s = α * g ** 2 + (1 - α) * state["s"]
        state.update({"s": s, "i": i})
        return η * torch.pow(i, ε - 0.5) / (τ + torch.sqrt(s))

    return ρ_generator
