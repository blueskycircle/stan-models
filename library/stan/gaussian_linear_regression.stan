data {
  int<lower=0> N; // Number of data points
  int<lower=0> K; // Number of predictors
  matrix[N, K] x; // Predictor matrix
  vector[N] y;    // Outcome vector
}

parameters {
  real alpha;           // Intercept
  vector[K] beta;       // Coefficients for predictors
  real<lower=0> sigma;  // Error standard deviation
}

model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 2.5);
  sigma ~ cauchy(0, 5);

  // Likelihood
  y ~ normal(x * beta + alpha, sigma);
}