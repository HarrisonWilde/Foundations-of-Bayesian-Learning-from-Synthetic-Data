data {

    int<lower=0> n;
    int<lower=0> n_groups;
    vector[n] observations;
    int school[n];
    real pmu;
    real psig;
    real<lower=0> w;
    real beta;
    real<lower=0> beta_w;

}

parameters {

    vector[n_groups] alphas;
    real mu;
    real<lower=0> sigy;
    real<lower=0> siga;

}

model {

    mu ~ normal(pmu, psig);
    sigy ~ inv_gamma(2, 5);
    siga ~ inv_gamma(2, 5);
    alphas ~ normal(mu, siga);
    for (i in 1:n) {
        observations[i] ~ normal(alphas[school[i]], sigy);
    }

}
