
# coding: utf-8

# # SV model with variational particle filter

# In[1]:


import matplotlib.pyplot as plt
import torch
import pandas as pd
pd.set_option('precision', 4)

import ptvi

dtype = torch.float64
nparticles = 10_000


# In[2]:


data_seed, algo_seed = 1234, 1234
params = dict(a=1., b=0., c=.95)
T = 200


# ## CPU

# In[3]:


model = ptvi.FilteredStochasticVolatilityModelFreeProposal(
    input_length=T, num_particles=nparticles, resample=True, dtype=torch.float64)
print(repr(model))


# In[4]:


torch.manual_seed(data_seed)
y, z_true = model.simulate(**params)


# In[6]:


torch.manual_seed(algo_seed)
trace = ptvi.PointEstimateTracer(model)
# ζ0 = torch.full((6,), 0.5, dtype=torch.float64)
fit = ptvi.stoch_opt(
    model, y, opt_type=torch.optim.Adamax,
    tracer=trace,
    stop_heur=ptvi.NullStoppingHeuristic(), 
    max_iters=2**8)

print(fit.summary(true=params))


# ## GPU
# 
# And now again with CUDA, if it's available

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')  # use the gpu if available


# In[ ]:


model = ptvi.FilteredStochasticVolatilityModelFreeProposal(
    input_length=T, num_particles=nparticles, resample=True, 
    device=device, dtype=dtype)
print(repr(model))


# In[ ]:


torch.manual_seed(data_seed)
y, z_true = model.simulate(**params)


# In[ ]:


torch.manual_seed(algo_seed)
trace = ptvi.PointEstimateTracer(model)
fit = ptvi.stoch_opt(model, y, opt_type=torch.optim.Adamax,
    tracer=trace, stop_heur=ptvi.NullStoppingHeuristic(), 
    max_iters=2**8)


# In[ ]:


print(fit.summary(true=params))

print('done')
