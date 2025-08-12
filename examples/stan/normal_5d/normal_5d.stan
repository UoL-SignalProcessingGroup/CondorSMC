data {
    int<lower=0> K;
    array[K] real df;
    array[K] real mu;
    real<lower=0, upper=1> phi;
}
parameters {
    array[K] real x;
}
model {
    for(k in 1:K) {
        target += phi * normal_lpdf(x[k] | mu[k], df[k]);
    }
}