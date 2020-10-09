// BetaD Logistic Regression
data {

    // Inputs for the sampler: data and prior hyperparameters
    int<lower=0> f;
    int<lower=0> a;
    matrix[a, f] X_real;
    int<lower=0, upper=1> y_real[a];
    int<lower=0> b;
    matrix[b, f] X_synth;
    int<lower=0, upper=1> y_synth[b];
    real<lower=0> w;
    real beta;
    real<lower=0> beta_w;
    int flag_real;
    int flag_synth;

}

parameters {

    // Parameters for which we do inference
    real alpha;
    vector[f] coefs;

}

transformed parameters {

    vector[b] logistic_xtheta;
    logistic_xtheta = inv_logit(alpha + X_synth * coefs);

}

model {

    // Uninformative priors
    alpha ~ normal(0, 50);
    coefs ~ normal(0, 50);

    // The likelihood
    if (flag_real == 0) {
        target += bernoulli_logit_glm_lpmf(y_real | X_real, alpha, coefs);
    }
    if (flag_synth == 0) {
        for (i in 1:b) {
            target += beta_w * (
                (1 / beta) * (logistic_xtheta[i] ^ y_synth[i] + (1 - logistic_xtheta[i]) ^ (1 - y_synth[i])) ^ (beta) -
                (1 / (beta + 1)) * (logistic_xtheta[i] ^ (beta + 1) + (1 - logistic_xtheta[i]) ^ (beta + 1))
            );
        }
    }

}


//
// generated quantities {
//
//     real log_like_test;
//     // vector[c] log_likes_test;
//     vector[c] probabilities_test;
//
//     log_like_test = bernoulli_logit_glm_lpmf(y_test | X_test, alpha, coefs);
//     // for (i in 1:c) {
//     //     log_likes_test[i] = bernoulli_logit_lpmf(y_test[i] | alpha + X_test[i] * coefs);
//     // }
//     probabilities_test = inv_logit(alpha + X_test * coefs);
//
// }
