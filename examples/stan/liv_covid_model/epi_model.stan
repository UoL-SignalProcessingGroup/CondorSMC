functions {
  array[] real seeiippd(real time,
                  array[] real state,
                  array[] real params,
                  array[] real real_data,
                  array[] int integer_data) {

    // Unpack integer data values
    int T = integer_data[1];
    int n_beta_pieces = integer_data[2];
    int n_disease_states = integer_data[3];

    // Unpack real data values
    array[n_beta_pieces] real beta_left_t = real_data[1:n_beta_pieces];
    array[n_beta_pieces] real beta_right_t = real_data[n_beta_pieces+1:2*n_beta_pieces];
    real population = real_data[2*n_beta_pieces+1];

    // Unpack parameter values
    array[n_beta_pieces] real beta_left = params[1:n_beta_pieces];
    array[n_beta_pieces] real grad_beta = params[n_beta_pieces+1:2*n_beta_pieces];
    real nu = params[2*n_beta_pieces+1];
    real gamma = params[2*n_beta_pieces+2];
    real kappa = params[2*n_beta_pieces+3];
    real omega = params[2*n_beta_pieces+4];

    // Unpack state
    real S = state[1];
    real E1 = state[2];
    real E2 = state[3];
    real I1 = state[4];
    real I2 = state[5];
    real P1 = state[6];
    real P2 = state[7];
    real D = state[8];

    real infection_rate;
    real nuE1 = nu * E1;
    real nuE2 = nu * E2;
    real gammaI1 = gamma * I1;
    real gammaI2 = gamma * I2;
    real kappaP1 = kappa * P1;
    real kappaP2 = kappa * P2;

    real dS_dt;
    real dE1_dt;
    real dE2_dt;
    real dI1_dt;
    real dI2_dt;
    real dP1_dt;
    real dP2_dt;
    real dD_dt;
    
    for (i in 1:n_beta_pieces) {
      if(time >= beta_left_t[i] && time < beta_right_t[i]) {
        real beta = grad_beta[i] * (time - beta_left_t[i]) + beta_left[i];
        infection_rate = beta * (I1 + I2) * S / population;
      }
    }

    dS_dt = -infection_rate;
    dE1_dt = infection_rate - nuE1;
    dE2_dt = nuE1 - nuE2;
    dI1_dt = nuE2 - gammaI1;
    dI2_dt = gammaI1 - gammaI2;
    dP1_dt = gammaI2 - kappaP1;
    dP2_dt = kappaP1 - kappaP2;
    dD_dt = kappaP2 * omega;

    return {dS_dt, dE1_dt, dE2_dt, dI1_dt, dI2_dt, dP1_dt, dP2_dt, dD_dt};
  }

  array[ , ] real integrate_ode_explicit_trapezoidal(array[] real initial_state, real initial_time, array[] real times, array[] real params, array[] real real_data, array[] int integer_data) {
    real h;
    vector[size(initial_state)] dstate_dt_initial_time;
    vector[size(initial_state)] dstate_dt_tidx;
    vector[size(initial_state)] k;
    array[size(times),size(initial_state)] real state_estimate;

    h = times[1] - initial_time;
    dstate_dt_initial_time = to_vector(seeiippd(initial_time, initial_state, params, real_data, integer_data));
    k = h*dstate_dt_initial_time;
    state_estimate[1,] = to_array_1d(to_vector(initial_state) + h*(dstate_dt_initial_time + to_vector(seeiippd(times[1], to_array_1d(to_vector(initial_state)+k), params, real_data, integer_data)))/2);

    for (tidx in 1:size(times)-1) {
      h = (times[tidx+1] - times[tidx]);
      dstate_dt_tidx = to_vector(seeiippd(times[tidx], state_estimate[tidx], params, real_data, integer_data));
      k = h*dstate_dt_tidx;
      state_estimate[tidx+1,] = to_array_1d(to_vector(state_estimate[tidx,]) + h*(dstate_dt_tidx + to_vector(seeiippd(times[tidx+1], to_array_1d(to_vector(state_estimate[tidx,])+k), params, real_data, integer_data)))/2);
    }

    return state_estimate;
  }
}
data {
  real initial_time;
  int<lower=1> n_beta_pieces;
  array[n_beta_pieces] real<lower=0> beta_left_t;
  array[n_beta_pieces] real<lower=0> beta_right_t;
  int<lower=1> n_rho_calls_111_pieces;
  array[n_rho_calls_111_pieces] int<lower=0> rho_calls_111_left_t;
  array[n_rho_calls_111_pieces] int<lower=0> rho_calls_111_right_t;
  int<lower=1> n_rho_hospital_admissions_pieces;
  array[n_rho_hospital_admissions_pieces] int<lower=0> rho_hospital_admissions_left_t;
  array[n_rho_hospital_admissions_pieces] int<lower=0> rho_hospital_admissions_right_t;
  int<lower=1> T;
  array[T] real times;
  int<lower=1> n_disease_states;
  real<lower=0> population;
  int<lower=1> deaths_length;
  int<lower=1> deaths_start;
  array[deaths_length] int<lower=0> deaths;
  int<lower=1> hospital_admissions_length;
  int<lower=1> hospital_admissions_start;
  array[hospital_admissions_length] int<lower=0> hospital_admissions;
  int<lower=1> calls_111_length;
  int<lower=1> calls_111_start;
  array[calls_111_length] int<lower=0> calls_111;
  int real_data_length;
  array[real_data_length] real real_data;
  int integer_data_length;
  array[integer_data_length] int integer_data;
}
transformed data {
  real mu_dL = 4.00;
  real sigma_dL = 3.00;
  real mu_dI = 5.00;
  real sigma_dI = 4.00;
  real mu_dT = 13.00;
  real sigma_dT = 4.00;
}
parameters {
  array[2] real<lower=0, upper=1> initial_state_raw;
  array[n_beta_pieces+1] real<lower=0> beta;
  real<lower=0> dL;
  real<lower=0> dI;
  real<lower=0> dT;
  real<lower=0, upper=1> omega;
  real<lower=0> reciprocal_phi_deaths;
  real<lower=0> reciprocal_phi_hospital_admissions;
  real<lower=0> reciprocal_phi_calls_111;
  array[n_rho_hospital_admissions_pieces+1] real<lower=0, upper=1> rho_hospital_admissions;
  array[n_rho_calls_111_pieces+1] real<lower=0, upper=1> rho_calls_111;
}
transformed parameters {
  array[n_disease_states] real initial_state;
  array[n_beta_pieces] real beta_left;
  array[n_beta_pieces] real beta_right;
  array[n_beta_pieces] real grad_beta;
  real nu;
  real gamma;
  real kappa;
  real phi_deaths;
  real phi_hospital_admissions;
  real phi_calls_111;
  array[n_rho_hospital_admissions_pieces] real rho_hospital_admissions_left;
  array[n_rho_hospital_admissions_pieces] real rho_hospital_admissions_right;
  array[n_rho_hospital_admissions_pieces] real grad_rho_hospital_admissions;
  array[n_rho_calls_111_pieces] real rho_calls_111_left;
  array[n_rho_calls_111_pieces] real rho_calls_111_right;
  array[n_rho_calls_111_pieces] real grad_rho_calls_111;
  array[T,n_disease_states] real state_estimate;
  vector[T+1] S;
  vector[T+1] E1;
  vector[T+1] E2;
  vector[T+1] I1;
  vector[T+1] I2;
  vector[T+1] P1;
  vector[T+1] P2;
  vector[T+1] D;
  vector[T] daily_infections;
  vector[T] daily_new_infectious;
  vector[T] daily_new_pending;
  vector[T] daily_deaths;
  vector[T] daily_hospital_admissions;
  vector[T] daily_calls_111;

  initial_state[1] = (population-5.0)*initial_state_raw[1] + 1.0;
  initial_state[2] = (population-5.0)*(1.0-initial_state_raw[1])*initial_state_raw[2]/2.0 + 1.0;
  initial_state[3] = (population-5.0)*(1.0-initial_state_raw[1])*initial_state_raw[2]/2.0 + 1.0;
  initial_state[4] = (population-5.0)*(1.0-initial_state_raw[1])*(1.0-initial_state_raw[2])/2.0 + 1.0;
  initial_state[5] = (population-5.0)*(1.0-initial_state_raw[1])*(1.0-initial_state_raw[2])/2.0 + 1.0;
  initial_state[6] = 0.0;
  initial_state[7] = 0.0;
  initial_state[8] = 0.0;
  beta_left = beta[:n_beta_pieces];
  beta_right = beta[2:];
  grad_beta = to_array_1d((to_vector(beta_right) - to_vector(beta_left))./(to_vector(beta_right_t) - 
              to_vector(beta_left_t)));
  nu = 2.0/dL;
  gamma = 2.0/dI;
  kappa = 2.0/dT;
  phi_deaths = 1.0 / reciprocal_phi_deaths;
  phi_hospital_admissions = 1.0 / reciprocal_phi_hospital_admissions;
  phi_calls_111 = 1.0 / reciprocal_phi_calls_111;
  rho_hospital_admissions_left = rho_hospital_admissions[:n_rho_hospital_admissions_pieces];
  rho_hospital_admissions_right = rho_hospital_admissions[2:];
  grad_rho_hospital_admissions = to_array_1d((to_vector(rho_hospital_admissions_right) - to_vector(rho_hospital_admissions_left))./(to_vector(rho_hospital_admissions_right_t) - to_vector(rho_hospital_admissions_left_t)));
  rho_calls_111_left = rho_calls_111[:n_rho_calls_111_pieces];
  rho_calls_111_right = rho_calls_111[2:];
  grad_rho_calls_111 = to_array_1d((to_vector(rho_calls_111_right) - to_vector(rho_calls_111_left))./(to_vector(rho_calls_111_right_t) - to_vector(rho_calls_111_left_t)));

  {
    array[2*n_beta_pieces+4] real params;
    params[1:n_beta_pieces] = beta_left;
    params[n_beta_pieces+1:2*n_beta_pieces] = grad_beta;
    params[2*n_beta_pieces+1] = nu;
    params[2*n_beta_pieces+2] = gamma;
    params[2*n_beta_pieces+3] = kappa;
    params[2*n_beta_pieces+4] = omega;

    state_estimate = integrate_ode_explicit_trapezoidal(initial_state, initial_time, times, params, real_data, integer_data);
  }

  S = append_row(initial_state[1], to_vector(state_estimate[, 1]));
  E1 = append_row(initial_state[2], to_vector(state_estimate[, 2]));
  E2 = append_row(initial_state[3], to_vector(state_estimate[, 3]));
  I1 = append_row(initial_state[4], to_vector(state_estimate[, 4]));
  I2 = append_row(initial_state[5], to_vector(state_estimate[, 5]));
  P1 = append_row(initial_state[6], to_vector(state_estimate[, 6]));
  P2 = append_row(initial_state[7], to_vector(state_estimate[, 7]));
  D = append_row(initial_state[8], to_vector(state_estimate[, 8]));

  daily_infections = S[:T] - S[2:] + machine_precision();
  daily_new_infectious = nu * E2[:T] + machine_precision();
  daily_new_pending = gamma * I2[:T] + machine_precision();
  daily_deaths = D[2:] - D[:T] + machine_precision();

  daily_hospital_admissions = rep_vector(0.0, T);

  for (time in 1:T) {
    for (i in 1:n_rho_hospital_admissions_pieces) {
      if(time >= rho_hospital_admissions_left_t[i] && time < rho_hospital_admissions_right_t[i]) {
        real rho_hospital_admissions_t = grad_rho_hospital_admissions[i] * (time - rho_hospital_admissions_left_t[i]) + rho_hospital_admissions_left[i];
        daily_hospital_admissions[time] = daily_new_pending[time] * rho_hospital_admissions_t;
      }
    }
  }

  daily_calls_111 = rep_vector(0.0, T);

  for (time in 1:T) {
    for (i in 1:n_rho_calls_111_pieces) {
      if(time >= rho_calls_111_left_t[i] && time < rho_calls_111_right_t[i]) {
        real rho_calls_111_t = grad_rho_calls_111[i] * (time - rho_calls_111_left_t[i]) + rho_calls_111_left[i];
        daily_calls_111[time] = (daily_new_infectious[time] + daily_new_pending[time]) * rho_calls_111_t;
      }
    }
  }
}
model {
  initial_state_raw[1] ~ beta(5.0, 0.5);
  initial_state_raw[2] ~ beta(1.1, 1.1);
  beta[1] ~ normal(0, 0.5);
  beta[2:] ~ normal(beta[:n_beta_pieces], 0.01);
  dL ~ normal(mu_dL, sigma_dL);
  dI ~ normal(mu_dI, sigma_dI);
  dT ~ normal(mu_dT, sigma_dT);
  omega ~ beta(5.7, 624.1);
  reciprocal_phi_deaths ~ exponential(5);
  reciprocal_phi_hospital_admissions ~ exponential(5);
  reciprocal_phi_calls_111 ~ exponential(5);
  rho_hospital_admissions ~ beta(1.1, 1.1);
  rho_calls_111 ~ beta(1.1, 1.1);

  target += neg_binomial_2_lpmf(deaths | daily_deaths[deaths_start:deaths_start+deaths_length-1], phi_deaths);
  target += neg_binomial_2_lpmf(hospital_admissions | daily_hospital_admissions[hospital_admissions_start:hospital_admissions_start+hospital_admissions_length-1], phi_hospital_admissions);
  target += neg_binomial_2_lpmf(calls_111 | daily_calls_111[calls_111_start:calls_111_start+calls_111_length-1], phi_calls_111);
}
generated quantities {
  vector[T+1] I = I1 + I2;

  vector[T] effective_reproduction_number = (daily_infections./I[:T])*dI;

  vector[T-7] growth_rate = (log(daily_infections[8:]) - log(daily_infections[:T-7]))/7.0;

  array[deaths_length+7] int pred_deaths;

  pred_deaths = neg_binomial_2_rng(daily_deaths[deaths_start:deaths_start+deaths_length-1+7], phi_deaths);

  array[hospital_admissions_length] int pred_hospital_admissions;

  pred_hospital_admissions = neg_binomial_2_rng(daily_hospital_admissions[hospital_admissions_start:hospital_admissions_start+hospital_admissions_length-1], phi_hospital_admissions);

  array[calls_111_length] int pred_calls_111;

  pred_calls_111 = neg_binomial_2_rng(daily_calls_111[calls_111_start:calls_111_start+calls_111_length-1], phi_calls_111);
}