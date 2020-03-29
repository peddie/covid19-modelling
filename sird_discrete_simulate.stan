functions {
  // Compute dy/dt given y and parameters.  This function may assume
  // that delta entries for all past timesteps are available, equally
  // spaced apart by `dt`.  This allows for delay dynamics.
  vector fixed_time_sird_dynamics(real t, real dt,
                                  vector y,
                                  vector params,
                                  int population,
                                  int n_past,
                                  matrix past_deltas) {
    // Expand parameter vector
    real beta = params[1];
    real gamma = params[2];
    real zeta = params[3];

    // In this version of the SIRD model, we assume a constant country
    // population and compute S (susceptible population) by
    // subtracting all the rest of the populations from the total
    // population.  The reason for doing this is that then we can
    // avoid figuring out background birth and death rates.  I think
    // this is justified because:
    //
    //    1) the number of cases is much smaller than the total
    //       population of the country
    //
    //    2) the timescale of the infection in the country is much
    //       faster than the timescale of the country's background
    //       population dynamics
    //
    // We shall see.
    real denom = population - y[3];
    real s = population - y[1] - y[2] - y[3];
    // beta * S / (N0 - D) * I
    real infections = beta * s * y[1] / denom;

    // Output vector
    vector[3] dydt;
    // dI/dt = beta * S / (N - D) * I - gamma * I - zeta * I
    dydt[1] = infections - gamma * y[1] - zeta * y[1];
    // dR/dt = gamma * I
    dydt[2] = gamma * y[1];
    // dD/dt = zeta * I
    dydt[3] = zeta * y[1];

    return dydt;
  }

  // Midpoint integration for now.  More efficient than Euler and
  // still has a fixed timestep.
  matrix midpoint_integrate_sird(int n_steps,
                                 real dt0,
                                 real[] y0,
                                 real[] params,
                                 int population) {
    vector[3] y = to_vector(y0);
    matrix[n_steps, 3] delta_history;
    for (k in 1:n_steps) {
      real t0 = dt0 * k;
      vector[3] dydt0 = fixed_time_sird_dynamics(t0, dt0, y, to_vector(params),
                                                 population, k - 1,
                                                 delta_history);
      real thalf = dt0 * (k + 0.5);
      vector[3] yhalf = y + dydt0 * dt0 * 0.5;
      vector[3] dydthalf = fixed_time_sird_dynamics(thalf, dt0, yhalf,
                                                    to_vector(params),
                                                    population, k - 1,
                                                    delta_history);
      vector[3] delta1 = dydthalf * dt0;
      delta_history[k, :] = to_row_vector(delta1);
      y = y + delta1;
    }

    return delta_history;
  }

  matrix accumulate_deltas(matrix deltas) {
    matrix[rows(deltas), cols(deltas)] totals;
    for (i in 1:cols(deltas)) {
      totals[:, i] = cumulative_sum(deltas[:, i]);
    }

    return totals;
  }
}

data {
  // Number of days observed
  int<lower=1> T;
  // Measurements of infected, recovered, dead
  real y0[3];
  // Time values (consecutive in practice)
  real ts[T];
  // Total population of the country
  int<lower=1> population;

  // Prior info
  real sigma_infected;
  real sigma_dead;
  real beta;
  real gamma;
  real zeta;
}
transformed data {
  real x_r[0];
  int x_i[1] = {population};
}
parameters {
}
model {
}

generated quantities {
  matrix[T, 3] log_likelihood;
  matrix[T, 3] dy_hat;
  real theta[3] = {beta, gamma, zeta};
  dy_hat[1] = to_row_vector(y0);
  dy_hat[2:T] = midpoint_integrate_sird(T - 1, 1, y0, theta, population);
  real y_hat[T, 3] = to_array_2d(accumulate_deltas(dy_hat));
  for (t in 1:T) {
    y_hat[t, 1] += normal_rng(0, sigma_infected);
    y_hat[t, 2] += normal_rng(0, sigma_dead);
    y_hat[t, 3] += normal_rng(0, sigma_dead);
  }
}
