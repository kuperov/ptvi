library(tidyverse)
library(stochvol)

# True parameters assumed to be a=1.0, b=0.0, c=0.95
# Drawing 100 variates from p(y_T+1, z_T+1 | z_T, a, b, c) with data_seed=1
# Stochastic volatility model:
#   x_t = exp(a * z_t/2) ε_t      t=1, …, 100
#   z_t = b + c * z_{t-1} + ν_t,  t=2, …, 100
#   z_1 = b + 1/√(1 - c^2) ν_1
#         where ε_t, ν_t ~ Ν(0,1)
# 
# Particle filter with 50 particles, AR(1) proposal:
#   z_t = b + c * z_{t-1} + η_t,  t=2, …, 100
#   z_1 = b + 1/√(1 - c^2) η_1
#         where η_t ~ Ν(0,1)

alldata <- read_csv('experiment.csv', 
             col_types = cols(
                t = col_integer(),
                y = col_double(),
                z = col_double()
              )
             )

y <- alldata[1:100,][['y']]

fit <- svsample(y, designmatrix = 'ar1')
summary(fit, showlatent = FALSE)
volplot(fit, forecast=10)

N <- 100

# produce N forecasts

