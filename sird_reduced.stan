functions {
  real[] sird_dynamics(real t,
                       real[] y,
                       real[] theta,
                       real[] x_r,
                       int[] x_i
                       ) {
    // Expand parameter vector
    real beta = theta[1];
    real gamma = theta[2];
    real zeta = theta[3];
    int population = x_i[1];

    real denom = population - y[3];
    real s = population - y[1] - y[2] - y[3];
    // beta * S / (N0 - D) * I
    real infections = beta * s * y[1] / denom;

    // Output vector
    real dydt[3];
    // dI/dt = beta * S / (N - D) * I - gamma * I - zeta * I
    dydt[1] = infections - gamma * y[1] - zeta * y[1];
    // dR/dt = gamma * I
    dydt[2] = gamma * y[1];
    // dD/dt = zeta * I
    dydt[3] = zeta * y[1];
    return dydt;
  }
}
data {
  int<lower=1> T;
  real y[T,3];
  real ts[T];
  int<lower=1> population;
}
transformed data {
  real x_r[0];
  int x_i[1] = {population};
}
parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> zeta;
  // real<lower=0, upper=1> theta[3];
  real<lower=0> sigma;
}
model {
  real y_hat[T,3];
  sigma ~ inv_gamma(1, 1);
  beta ~ inv_gamma(2.2, 0.15);
  gamma ~ inv_gamma(3, 0.12);
  zeta ~ inv_gamma(3.5, 0.02);
  real theta[3] = {beta, gamma, zeta};
  y_hat[1] = y[1];
  y_hat[2:T] = integrate_ode_rk45(sird_dynamics, y[1], ts[1], ts[2:T], theta, x_r, x_i);
  for (t in 1:T)
    y[t] ~ normal(y_hat[t], sigma);
}

generated quantities {
  matrix[T, 3] log_likelihood;
  real y_hat[T, 3];
  real theta[3] = {beta, gamma, zeta};
  y_hat[1] = y[1];
  y_hat[2:T] = integrate_ode_rk45(sird_dynamics, y[1], ts[1], ts[2:T], theta, x_r, x_i);
  for (t in 1:T)
    for (i in 1:3)
      log_likelihood[t, i] = normal_lpdf(y_hat[t, i] | y[t, i], sigma);
}
