import pickle
import os


def cache_stan_model(stanfile):
    """Load and compile the Stan model.

    This function can take a while the first time it is run. The returned model object
    is reusable for multiple inference runs, including with different data and hyper-
    parameters.

    Args:
        stanfile:  stan file (must be in the current directory)
    
    Returns:
        StanModel object
    """
    import pystan as ps  # imported locally in case Stan is borked
    stan_abs = os.path.join(os.path.dirname(__file__), stanfile)
    pickle_file = os.path.join(os.path.dirname(__file__), f"{stanfile}.pkl")
    mdl_name = stanfile.replace('.', '_')
    if os.path.isfile(pickle_file):
        print(f'Re-using pickled {mdl_name} in {pickle_file}.')
        with open(pickle_file, "rb") as fp:
            sm = pickle.load(fp)
    else:
        sm = ps.StanModel(file=stan_abs, model_name=mdl_name)
        with open(pickle_file, "wb") as fp:
            pickle.dump(sm, fp)
        print(f'Caching model {mdl_name} to {pickle_file}.')
    return sm
