data {
  int<lower=0> N;
  int<lower=0> k;
  int<lower=0,upper=1> y[N];
  matrix[N, k] X;
  vector[k] mu_beta;
  matrix[k, k] Sigma_beta;
}
parameters {
  vector[k] beta;
}
model {
  beta ~ multi_normal(mu_beta, Sigma_beta);
  y ~ bernoulli(Phi(X*beta));
}