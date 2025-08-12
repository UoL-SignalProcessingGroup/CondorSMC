data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
  real<lower=0, upper=1> phi;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta_tilde;
}
transformed parameters {
  array[J] real theta;
  for (j in 1 : J) 
    theta[j] = mu + tau * theta_tilde[j];
}
model {
  //mu ~ normal(0, 5);
  //tau ~ cauchy(0, 5);
  //theta_tilde ~ normal(0, 1);

  target += normal_lpdf(theta_tilde | 0, 1);
  target += normal_lpdf(mu | 0, 5);
  target += cauchy_lpdf(tau | 0, 5);

//  y ~ normal(theta, sigma);
  target += phi * normal_lpdf(y | theta, sigma);
}