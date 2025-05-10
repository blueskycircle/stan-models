data {
  int<lower=0> N; // Number of data points
  int<lower=0> K; // Number of predictors
  matrix[N, K] x; // Predictor matrix
  vector[N] y;    // Outcome vector
}

transformed data {
  // Perform QR decomposition on the predictor matrix x
  // Q_ast is an N x K matrix with orthogonal columns
  // R_ast is a K x K upper triangular matrix
  matrix[N, K] Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  matrix[K, K] R_ast = qr_thin_R(x) / sqrt(N - 1);
}

parameters {
  real alpha;           // Intercept
  vector[K] theta;      // Coefficients for Q_ast (transformed parameters)
  real<lower=0> sigma;  // Error standard deviation
}

model {
  // Priors
  alpha ~ normal(0, 10);
  theta ~ normal(0, 1);
  sigma ~ cauchy(0, 5);  

  // Likelihood
  // y ~ normal(Q_ast * theta + alpha, sigma)
  // This is equivalent to y ~ normal(x * beta + alpha, sigma)
  // where beta = R_ast_inverse * theta
  y ~ normal(Q_ast * theta + alpha, sigma);
}

generated quantities {
  vector[K] beta; // Original predictor coefficients

  // Recover beta from theta: R_ast * beta = theta  => beta = inverse(R_ast) * theta
  matrix[K, K] R_ast_inv = inverse(R_ast);
  beta = R_ast_inv * theta;
}