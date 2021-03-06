set.seed(5)
#
# mu <- 0
# sigma2_alpha <- 1
# sigma2_y <- 1.5
#
# n_groups <- 3
# n <- 1200
#
# school <- c(rep(1, 400), rep(2, 400), rep(3, 400))
#
# alphas <- rnorm(n_groups, mu, sqrt(sigma2_alpha))
#
# obs <- rnorm(n, alphas[school], sqrt(sigma2_y))
#
# multilevel_normal_data <- list(
#     n = n,
#     n_groups = n_groups,
#     observations = obs,
#     school = school,
#     pmu = 0,
#     psig = 2,
#     w = 0.5,
#     beta = 0.5,
#     beta_w = 1.2
# )
#
# model_stan <- stan_model(file="src/regression/stan/simple1.stan")
# model <- sampling(
#     object=model_stan,
#     data=multilevel_normal_data,
#     iter=2 * 1000,
#     chains=1,
#     cores=1
# )
# params <- extract(model)


library(mlmRev)
library(rstan)
library(rio)
library(dplyr)
data(Gcsemv, package = "mlmRev")
Gcsemv$female <- relevel(Gcsemv$gender, "M")
GCSE <- subset(x = Gcsemv, select = c(school, student, female, course))
GCSE <- GCSE %>% mutate(female = ifelse(female == "M", 0, 1)) %>% na.omit()
GCSE %>% export("gcse.csv")
# hierarchical_data <- list(
#     n_observations = nrow(GCSE),
#     n_groups = length(unique(GCSE$school)),
#     p_theta = 1,
#     y = GCSE$course,
#     X = GCSE$female,
#     schools = GCSE$school,
#     p_mu = 0,
#     p_sigma = 1.5,
#     p_alpha = 2,
#     p_beta = 5,
#     p_nu = 2,
#     p_Sigma = ,
#     w = 0.5,
#     beta = 0.5,
#     beta_w = 1.25,
# )


mu <- 1


n_observations = 1200
n_groups = 3
n_theta = 1
y = rnorm(n, alphas[school], sqrt(sigma2_y))
X = GCSE$female
schools = GCSE$school
p_mu = 0
p_sigma = 1.5
p_alpha = 2
p_beta = 5
p_nu = 2
p_Sigma = rep(3, n_theta)
w = 0.5
beta = 0.5
beta_w = 1.25

hierarchical_data <- list(
    n_observations = n_observations,
    n_groups = n_groups,
    n_theta = n_theta,
    y = y,
    X = X,
    schools = schools,
    p_mu = p_mu,
    p_sigma = p_sigma,
    p_alpha = p_alpha,
    p_beta = p_beta,
    p_nu = p_nu,
    p_Sigma = p_Sigma,
    w = w,
    beta = beta,
    beta_w = beta_w,
)

hier_model_stan <- stan_model(file="src/regression/stan/model.stan")
model <- sampling(
    object=hier_model_stan,
    data=hierarchical_data,
    iter=2 * 1000,
    chains=1,
    cores=1,
)
params <- extract(model)
