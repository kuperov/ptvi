// Simple hierarchical Bayesian regression.
// Model is Drugowitsch's non-ARD VB model in https://arxiv.org/abs/1310.5438.
data {
  int<lower=0> N;
  int<lower=0> k;
  vector[N] y;
  matrix[N, k] X;
  real<lower=0> a_0;         // tau shape hyperparameter
  real<lower=0> b_0;         // tau rate hyperparameter
  real<lower=0> c_0;         // alpha shape hyperparameter
  real<lower=0> d_0;         // alpha rate hyperparameter
}
parameters {
  vector[k] beta;           // predictors
  real<lower=0> tau;        // noise precision
  real<lower=0> alpha;      // common predictor prior precision
}
model {
  // priors
  tau ~ gamma(a_0, b_0);
  alpha ~ gamma(c_0, d_0);
  beta ~ normal(0, 1/sqrt(alpha * tau));
  // likelihood
  y ~ normal(X * beta, 1/sqrt(tau));
}
