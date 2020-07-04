data {

    int<lower=0> n_real;
    int<lower=0> n_synth;
    int<lower=0> n_groups;
    int<lower=0> n_theta;
    vector[n_real] y_real;
    vector[n_synth] y_synth;
    matrix[n_real, n_theta] X_real;
    matrix[n_synth, n_theta] X_synth;
    int schools_real[n_real];
    int schools_synth[n_synth];
    real p_mu;
    real p_sigma;
    real p_alpha;
    real p_beta;
    real p_nu;
    cov_matrix[n_theta] p_Sigma;
    real<lower=0> w;
    real beta;
    real<lower=0> beta_w;
    int flag_real;
    int flag_synth;

}

parameters {

    vector[n_theta] Theta[n_groups];
    vector[n_theta] mu_theta;
    cov_matrix[n_theta] Sigma;
    //real<lower=0> sigma_y;
    vector<lower=0>[n_groups] sigma2;
    real<lower=0> alpha_sigma2;
    real<lower=0> beta_sigma2;

}

transformed parameters {

    vector[n_real] lin_pred_real;
    vector[n_synth] lin_pred_synth;
    if (flag_real == 0) {
        for (i in 1:n_real) {
            lin_pred_real[i] = dot_product(X_real[i], Theta[schools_real[i]]);
        }
    }
    if (flag_synth == 0) {
        for (i in 1:n_synth) {
            lin_pred_synth[i] = dot_product(X_synth[i], Theta[schools_synth[i]]);
        }
    }

}

model {

    alpha_sigma2 ~ inv_gamma(p_alpha, p_beta);
    beta_sigma2 ~ inv_gamma(p_alpha, p_beta);
    mu_theta ~ normal(p_mu, p_sigma);
    Sigma ~ inv_wishart(p_nu, p_Sigma);
    // sigma2 ~ inv_gamma(p_alpha, p_beta)
    if (flag_real + flag_synth < 2) {
        for (j in 1:n_groups) {
            Theta[j] ~ multi_normal(mu_theta, Sigma);
            sigma2[j] ~ inv_gamma(alpha_sigma2, beta_sigma2);
        }
    }

    if (flag_real == 0) {
        for (i in 1:n_real) {
            y_real[i] ~ normal(lin_pred_real[i], sqrt(sigma2[schools_real[i]]));
        }
    }
    if (flag_synth == 0) {
        for (i in 1:n_synth) {
            y_synth[i] ~ normal(lin_pred_synth[i], sqrt(sigma2[schools_synth[i]]));
        }
    }

}
