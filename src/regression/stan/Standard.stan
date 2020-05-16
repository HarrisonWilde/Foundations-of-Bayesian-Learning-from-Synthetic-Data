data {

}

parameters {

}

model {

    alpha ~ normal(0, 100);
    coefs ~ normal(0, 100);
    course ~ female + (1 | school)
}
