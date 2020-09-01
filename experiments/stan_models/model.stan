data {

    int<lower=0> n_observations;
    int<lower=0> n_groups;
    int<lower=0> n_theta;
    vector[n_observations] y;
    matrix[n_observations, n_theta] X;
    int schools[n_observations];
    real p_mu;
    real p_sigma;
    real p_alpha;
    real p_beta;
    real p_nu;
    cov_matrix[n_theta] p_Sigma;
    real<lower=0> w;
    real beta;
    real<lower=0> beta_w;

}

parameters {

    matrix[n_groups, n_theta] Theta;
    vector[n_theta] mu_theta;
    real<lower=0> sigma_y;
    cov_matrix[n_theta] Sigma[n_groups];

}

transformed parameters {

    vector[n] lin_pred;
    for (i in 1:n) {
        lin_pred[i] = X[i,] * Theta[schools[i],];
    }

}

model {

    mu_theta ~ normal(p_mu, p_sigma);
    sigma_y ~ inv_gamma(p_alpha, p_beta);
    Sigma ~ inv_wishart(p_nu, p_Sigma);

    for (j in 1:n_groups) {
        Theta[j,] ~ multi_normal(mu_theta[j], Sigma[j]);
    }

    for (i in 1:n) {
        y[i] ~ normal(lin_pred[i], sigma_y);
    }

}
