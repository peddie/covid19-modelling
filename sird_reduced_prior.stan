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

    // In this version of the SIRD model, we assume a constant country
    // population and compute S (susceptible population) by subtracting all the
    // rest of the populations from the total population.  The reason for doing
    // this is that then we can avoid figuring out background birth and death
    // rates.  I think this is justified because:
    //
    //    1) the number of cases is much smaller than the total population of the country
    //
    //    2) the timescale of the infection in the country is much faster than
    //       the timescale of the country's background population dynamics
    //
    // We shall see.
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
  // Number of days observed
  int<lower=1> T;
  // Measurements of infected, recovered, dead
  real y[T, 3];
  // Time values (consecutive in practice)
  real ts[T];
  // Total population of the country
  int<lower=1> population;

  // Prior info
  real sigma_infected_prior[2];
  real sigma_dead_prior[2];
  real beta_prior[2];
  real gamma_prior[2];
  real zeta_prior[2];
}
transformed data {
  real x_r[0];
  int x_i[1] = {population};
}
parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=0> zeta;
  real<lower=0> sigma_infected;
  real<lower=0> sigma_dead;
}
model {
  // Declare priors
  sigma_infected ~ inv_gamma(sigma_infected_prior[1], sigma_infected_prior[2]);
  sigma_dead ~ inv_gamma(sigma_dead_prior[1], sigma_dead_prior[2]);
  beta ~ inv_gamma(beta_prior[1], beta_prior[2]);
  gamma ~ normal(gamma_prior[1], gamma_prior[2]);
  zeta ~ normal(zeta_prior[1], zeta_prior[2]);
}

generated quantities {
  matrix[T, 3] log_likelihood;
  real y_hat[T, 3];
  real theta[3] = {beta, gamma, zeta};
  y_hat[1] = y[1];
  y_hat[2:T] = integrate_ode_rk45(sird_dynamics, y_hat[1], ts[1], ts[2:T], theta, x_r, x_i);
  for (t in 1:T) {
    log_likelihood[t, 1] = normal_lpdf(y[t, 1] | y_hat[t, 1], fabs(y_hat[t, 1]) * sigma_infected);
    log_likelihood[t, 2] = normal_lpdf(y[t, 2] | y_hat[t, 2], (y_hat[t, 2] + 1) * sigma_dead);
    log_likelihood[t, 3] = normal_lpdf(y[t, 3] | y_hat[t, 3], (y_hat[t, 3] + 1)* sigma_dead);
  }
}
