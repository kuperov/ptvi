data {
  int<lower=0> N;
  int<lower=0> k;
  int y[N];
  matrix[N, k] X;
  vector[k] mu_beta;
  matrix[k, k] Sigma_beta;
}
parameters {
  vector[k] beta;
}
model {
  beta ~ multi_normal(mu_beta, Sigma_beta);
  y ~ poisson(exp(X*beta));
}
