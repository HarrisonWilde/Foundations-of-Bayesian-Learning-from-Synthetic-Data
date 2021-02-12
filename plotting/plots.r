library(rio)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(Cairo)


format_metric <- function(str) {
    if (str == "kld") {
        return("Kullback-Leibler Distance")
    } else if (str == "wass") {
        return("Wasserstein Distance")
    } else if (str == "ll") {
        return("Log Score")
    } else if (str == "param_mse") {
        return("Total Parameter Posterior Squared Distance")
    } else {
        return("No Formatting Given for this Metric")
    }
}


resampled <- "#ffc107"
w0 <- "#ffab91"
w05 <- "#ff5722"
w1 <- "#d84315"
b025 <- "#90caf9"
b05 <- "#2196f3"
b075 <- "#1565c0"
noiseaware <- "#4caf50"


# path <- "from_cluster/gaussian/outputs/final_csvs/neff_final.csv"
path <- "from_cluster/gaussian/outputs/final_csvs/neff.csv"
# path <- "gaussian/outputs/noise_demo.csv"
metrics <- c("kld", "ll", "wass")
out_path <- str_remove(str_replace(paste(str_split(path, "/")[[1]][-4], collapse="/"), "outputs", "plots_new"), ".csv")
dir.create(out_path)

data <- drop_na(import(path, setclass="tibble", fill=TRUE), iter)




conf_interval <- 0.95


data <- data %>%
    mutate(model_full = ifelse(
        model == "noise_aware", "Noise-Aware", ifelse(
            model == "beta", paste0(intToUtf8(0x03B2), " = ", beta), ifelse(
                model == "resampled", "Resampled", ifelse(
                    ((model == "weighted") & (weight == 0)), "w = 0 (No Syn)", ifelse(
                        ((model == "weighted") & (weight == 1)), "w = 1 (Naive)", paste0("w = ", weight))))))) %>% 
    select(-c(model, beta, weight))

real_ns = c(2, 4, 6, 8, 10, 13, 16, 19, 22, 25, 30, 35, 40, 50, 75, 100)

data_without_reals <- data %>% filter(real_n %in% real_ns)






# N EFFECTIVE PLOTS

real_avgs <- data %>% 
    filter(synth_n == 0) %>% 
    group_by(real_n) %>% 
    summarise(
        matched_ll = mean(ll), matched_wass = mean(wass), matched_kld = mean(kld)
    ) %>%
    mutate(n_eff_ll = real_n, n_eff_kld = real_n, n_eff_wass = real_n) %>%
    select(-real_n)





# Bootstrap n effective plots

N <- 100
B <- 500

uniques <- data_without_reals %>% select(real_n, noise) %>% distinct()

disty <- function(x, y) abs(x - y)

data_list <- list()

for (j in 1:nrow(uniques)) {
    u <- uniques[j,]
    n1 <- u[[1]]
    n2 <- u[[2]]
    working_df <- data_without_reals %>% filter(real_n == n1, noise == n2) %>% arrange(model_full)
    seeds <- unique(working_df$seed)
    b_list <- list()
    for (i in 1:B) {
        b_seeds <- tibble(seed=sample(seeds, N, replace=TRUE))
        b_df <- b_seeds %>%
            inner_join(working_df) %>%
            group_by(synth_n, model_full) %>%
            summarise(b_exp_ll = mean(ll), b_exp_kld = mean(kld), b_exp_wass = mean(wass), .groups="drop") %>%
            group_by(model_full) %>%
            summarise(
                b_min_exp_ll = min(b_exp_ll), b_min_exp_kld = min(b_exp_kld), b_min_exp_wass = min(b_exp_wass),
                n_min_exp_ll = synth_n[which.min(b_exp_ll)], n_min_exp_kld = synth_n[which.min(b_exp_kld)], n_min_exp_wass = synth_n[which.min(b_exp_wass)], .groups="drop"
            )

        idx <- sapply(b_df$b_min_exp_ll, function(x) which.min( disty(x, real_avgs$matched_ll)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_ll, n_eff_ll))[idx,1:2,drop=FALSE])
        idx <- sapply(b_df$b_min_exp_kld, function(x) which.min( disty(x, real_avgs$matched_kld)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_kld, n_eff_kld))[idx,1:2,drop=FALSE])
        idx <- sapply(b_df$b_min_exp_wass, function(x) which.min( disty(x, real_avgs$matched_wass)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_wass, n_eff_wass))[idx,1:2,drop=FALSE])
        b_df <- b_df %>% mutate(
            n_eff_ll = n_eff_ll - n1,
            n_eff_kld = n_eff_kld - n1,
            n_eff_wass = n_eff_wass - n1
        )
        b_list[[i]] <- b_df
    }
    working_df <- do.call(rbind, b_list)
    working_df <- working_df %>%
        group_by(model_full) %>%
        summarise(
            exp_n_eff_ll = mean(n_eff_ll),
            exp_n_eff_kld = mean(n_eff_kld),
            exp_n_eff_wass = mean(n_eff_wass),
            se_n_eff_ll = sd(n_eff_ll),
            se_n_eff_kld = sd(n_eff_kld),
            se_n_eff_wass = sd(n_eff_wass)
        )
    working_df <- bind_cols(
        tibble(real_n = rep(n1, nrow(working_df)), noise = rep(n2, nrow(working_df))),
        working_df
    )
    data_list[[j]] <- working_df
}

n_eff_plot_df <- do.call(rbind, data_list)
noises <- n_eff_plot_df %>% pull(noise) %>% unique()
n_eff_plot_df <- n_eff_plot_df %>% filter(!model_full %in% c("Noise-Aware", "w = 0 (No Syn)"))

for (noise in noises) {
    for (metric in metrics) {
        n_eff_plot_df <- n_eff_plot_df %>% mutate(
            exp_metric = pmax(0, !!sym(paste0("exp_n_eff_", metric))),
            ci_low_metric = pmax(0, !!sym(paste0("exp_n_eff_", metric))) - pmin(2, !!(sym(paste0("se_n_eff_", metric))) / 4),
            ci_high_metric = pmax(0, !!sym(paste0("exp_n_eff_", metric))) + pmin(2, !!(sym(paste0("se_n_eff_", metric))) / 4),
        )
        p <- ggplot(n_eff_plot_df %>% filter(!!noise == noise), aes(x=real_n, y=exp_metric, color=model_full)) +
            geom_errorbar(
                aes(ymin=ci_low_metric, ymax=ci_high_metric),
                alpha = 0.5
            ) +
            geom_line() +
            labs(x = "Number of Real Samples", y = "Maximum Effective Real Sample Gain\nThroough the Use of Synthetic Data", color = "Model Type") +
            scale_color_manual(values=c(w05, w1, b05, b075)) +
            theme_light()
        ggsave(paste0(out_path, "/n_eff__noise_", noise, "__metric_", metric, ".pdf"), p, dpi=320, height=4, width=7, device=cairo_pdf)

    }
}

for (noise in noises) {
    df1 <- n_eff_plot_df %>% mutate(
        metric = format_metric("ll"),
        exp_metric = pmax(0, !!sym(paste0("exp_n_eff_ll"))),
        ci_low_metric = pmax(0, !!sym(paste0("exp_n_eff_ll"))) - pmin(2, !!(sym(paste0("se_n_eff_ll"))) / 4),
        ci_high_metric = pmax(0, !!sym(paste0("exp_n_eff_ll"))) + pmin(2, !!(sym(paste0("se_n_eff_ll"))) / 4),
    )
    df2 <- bind_rows(
        df1,
        n_eff_plot_df %>% mutate(
            metric = format_metric("kld"),
            exp_metric = pmax(0, !!sym(paste0("exp_n_eff_kld"))),
            ci_low_metric = pmax(0, !!sym(paste0("exp_n_eff_kld"))) - pmin(2, !!(sym(paste0("se_n_eff_kld"))) / 4),
            ci_high_metric = pmax(0, !!sym(paste0("exp_n_eff_kld"))) + pmin(2, !!(sym(paste0("se_n_eff_kld"))) / 4),
        )
    )
    df3 <- bind_rows(
        df2,
        n_eff_plot_df %>% mutate(
            metric = format_metric("wass"),
            exp_metric = pmax(0, !!sym(paste0("exp_n_eff_wass"))),
            ci_low_metric = pmax(0, !!sym(paste0("exp_n_eff_wass"))) - pmin(2, !!(sym(paste0("se_n_eff_wass"))) / 4),
            ci_high_metric = pmax(0, !!sym(paste0("exp_n_eff_wass"))) + pmin(2, !!(sym(paste0("se_n_eff_wass"))) / 4),
        )
    )
    p <- ggplot(df3 %>% filter(!!noise == noise), aes(x=real_n, y=exp_metric, color=model_full)) +
        facet_wrap(. ~ metric) +
        geom_errorbar(
            aes(ymin=ci_low_metric, ymax=ci_high_metric),
            alpha = 0.5
        ) +
        geom_line() +
        labs(x = "Number of Real Samples", y = "Maximum Effective Real Sample Gain\nThroough the Use of Synthetic Data", color = "Model Type") +
        scale_color_manual(values=c(w05, w1, b05, b075)) +
        theme_light()
    ggsave(paste0(out_path, "/n_eff_joined___noise_", noise, ".pdf"), p, dpi=320, height=4, width=14, device=cairo_pdf)

}


# Expectation comparison plots

exp_data <- data_without_reals %>%
    group_by(real_n, synth_n, model_full, noise) %>%
    summarise(
        exp_ll = mean(ll),
        exp_kld = mean(kld),
        exp_wass = mean(wass),
        .groups="drop"
    )
min_exp_data <- exp_data %>%
    group_by(real_n, model_full, noise) %>%
    arrange(synth_n) %>%
    summarise(
        min_exp_ll = min(exp_ll),
        min_exp_kld = min(exp_kld),
        min_exp_wass = min(exp_wass),
        min_exp_synth_n_ll = synth_n[which.min(exp_ll)], 
        min_exp_synth_n_kld = synth_n[which.min(exp_kld)], 
        min_exp_synth_n_wass = synth_n[which.min(exp_wass)], 
        .groups="drop"
    )

disty <- function(x, y) abs(x - y)
idx <- sapply(min_exp_data$min_exp_ll, function(x) which.min( disty(x, real_avgs$matched_ll)))
exp_neff_df <- bind_cols(min_exp_data, select(real_avgs, c(matched_ll, n_eff_ll))[idx,1:2,drop=FALSE])
idx <- sapply(exp_neff_df$min_exp_kld, function(x) which.min( disty(x, real_avgs$matched_kld)))
exp_neff_df <- bind_cols(exp_neff_df, select(real_avgs, c(matched_kld, n_eff_kld))[idx,1:2,drop=FALSE])
idx <- sapply(exp_neff_df$min_exp_wass, function(x) which.min( disty(x, real_avgs$matched_wass)))
exp_neff_df <- bind_cols(exp_neff_df, select(real_avgs, c(matched_wass, n_eff_wass))[idx,1:2,drop=FALSE])

exp_neff_df <- exp_neff_df %>% mutate(
    n_eff_ll = n_eff_ll - real_n,
    n_eff_kld = n_eff_kld - real_n,
    n_eff_wass = n_eff_wass - real_n
)

noises <- exp_neff_df %>% pull(noise) %>% unique()

for (noise in noises) {
    for (metric in metrics) {
        exp_neff_df <- exp_neff_df %>% mutate(
            exp_metric = !!sym(paste0("n_eff_", metric)),
            # ci_low_metric = !!sym(paste0("n_eff_", metric)) - !!(sym(paste0("n_eff_", metric, "_se"))),
            # ci_high_metric = !!sym(paste0("n_eff_", metric)) + !!(sym(paste0("n_eff_", metric, "_se"))),
        )
        p <- ggplot(
            exp_neff_df %>% filter(!!noise == noise),
            aes(x=real_n, y=exp_metric, color=model_full)
        ) +
        # geom_errorbar(
        #     aes(ymin=ci_low_metric, ymax=ci_high_metric),
        #     alpha = 0.5
        # ) +
        geom_line() +
        labs(x = "Number of Real Samples", y = "Effective Real Samples to Gain Using Synthetic Data", color = "Model Type")
        ggsave(paste0(out_path, "/exp_n_eff__noise_", noise, "__metric_", metric, ".pdf"), p, device=cairo_pdf)
    }
}

# Bootstrap old and wrong


N <- 100
B <- 1000

min_data <- data_without_reals %>%
    group_by(real_n, seed, model_full, noise) %>%
    arrange(synth_n) %>%
    summarise(
        min_ll = min(ll),
        min_kld = min(kld), 
        min_wass = min(wass), 
        min_synth_n_ll = synth_n[which.min(ll)], 
        min_synth_n_kld = synth_n[which.min(kld)], 
        min_synth_n_wass = synth_n[which.min(wass)], 
        .groups="drop"
    )

disty <- function(x, y) abs(x - y)
idx <- sapply(min_data$min_ll, function(x) which.min( disty(x, real_avgs$matched_ll)))
neff_df <- bind_cols(min_data, select(real_avgs, c(matched_ll, n_eff_ll))[idx,1:2,drop=FALSE])
idx <- sapply(min_data$min_kld, function(x) which.min( disty(x, real_avgs$matched_kld)))
neff_df <- bind_cols(neff_df, select(real_avgs, c(matched_kld, n_eff_kld))[idx,1:2,drop=FALSE])
idx <- sapply(min_data$min_wass, function(x) which.min( disty(x, real_avgs$matched_wass)))
neff_df <- bind_cols(neff_df, select(real_avgs, c(matched_wass, n_eff_wass))[idx,1:2,drop=FALSE])

real_ns <- neff_df %>% pull(real_n) %>% unique()
neff_df <- neff_df %>% mutate(
    n_eff_ll = n_eff_ll - real_n,
    n_eff_kld = n_eff_kld - real_n,
    n_eff_wass = n_eff_wass - real_n
)
data_list <- list()

for (j in 1:length(real_ns)) {
    n <- real_ns[j]
    working_df <- neff_df %>% filter(real_n == n) %>% arrange(model_full, noise)
    left_side <- working_df %>% select(model_full, noise) %>% distinct()
    dims <- left_side %>% count() %>% pull()
    matrices <- array(0, c(dims, 3, B))
    for (i in 1:B) {
        matrices[,,i] <- working_df %>%
            group_by(model_full, noise) %>%
            sample_n(N) %>%
            arrange(model_full, noise) %>%
            summarise(b_n_eff_ll = mean(n_eff_ll), b_n_eff_kld = mean(n_eff_kld), b_n_eff_wass = mean(n_eff_wass), .groups="drop") %>%
            select(b_n_eff_ll, b_n_eff_kld, b_n_eff_wass) %>%
            as.matrix()
    }
    means <- apply(matrices, 1:2, mean)
    var <- array(0, c(dims, 3))
    for (i in 1:B) {
        var <- var + (matrices[,,i] - means) ^ 2
    }
    ses <- (var / B) ^ 0.5
    step <- bind_cols(
        left_side,
        bind_cols(
            tibble(real_n = rep(n, dims)),
            bind_cols(
                tibble(n_eff_ll = means[,1], n_eff_kld = means[,2], n_eff_wass = means[,3]),
                tibble(n_eff_ll_se = ses[,1], n_eff_kld_se = ses[,2], n_eff_wass_se = ses[,3])
            )
        )
    )
    data_list[[j]] <- step
}

n_eff_plot_df <- do.call(rbind, data_list)
noises <- n_eff_plot_df %>% pull(noise) %>% unique()

for (noise in noises) {
    for (metric in metrics) {
        n_eff_plot_df <- n_eff_plot_df %>% mutate(
            exp_metric = !!sym(paste0("n_eff_", metric)),
            ci_low_metric = !!sym(paste0("n_eff_", metric)) - !!(sym(paste0("n_eff_", metric, "_se"))),
            ci_high_metric = !!sym(paste0("n_eff_", metric)) + !!(sym(paste0("n_eff_", metric, "_se"))),
        )
        p <- ggplot(
            n_eff_plot_df %>% filter(!!noise == noise),
            aes(x=real_n, y=exp_metric, color=model_full)
        ) +
        geom_errorbar(
            aes(ymin=ci_low_metric, ymax=ci_high_metric),
            alpha = 0.5
        ) +
        geom_line() +
        labs(x = "Number of Real Samples", y = "Effective Real Samples to Gain Using Synthetic Data", color = "Model Type")
        ggsave(paste0(out_path, "/n_eff__noise_", noise, "__metric_", metric, ".pdf"), p, device=cairo_pdf)
    }
}






## The Normal-Laplace pdf ##
normal_laplace_pdf2 <- function(y, mu, sigma, lambda){
  return(
    1/(2*lambda)*exp((mu - y)/lambda + sigma^2/(2*lambda^2))*pnorm((y-mu)/sigma - sigma/lambda, 0, 1)
    +
      1/(2*lambda)*exp((y - mu)/lambda + sigma^2/(2*lambda^2))*(1-pnorm((y-mu)/sigma + sigma/lambda, 0, 1))
  )
}

## Numerical Integration for the KLD ##
KLD_univariate <- function(log_pdf_g, log_pdf_f){
  #KLD <- integrate(f = function(x){exp(log_pdf_g(x)) * (log_pdf_g(x) - log_pdf_f(x))}, lower = -Inf, upper = Inf)
  KLD1 <- integrate(f = function(x){exp(log_pdf_g(x)) * log_pdf_g(x)}, lower = -20, upper = 20)
  KLD2 <- integrate(f = function(x){exp(log_pdf_g(x)) * log_pdf_f(x)}, lower = -20, upper = 20)
  #return(KLD$value)
  return(KLD1$value - KLD2$value)
}

## Numerical Integration for the betaD ##
betaD_univariate <- function(pdf_g, pdf_f, beta){
  int_term_f <- integrate(f = function(x){pdf_f(x)^(1+beta)}, lower = -20, upper = 20)
  int_term_g <- integrate(f = function(x){pdf_g(x)^(1+beta)}, lower = -20, upper = 20)
  cross_term <- integrate(f = function(x){pdf_g(x) * pdf_f(x)^beta}, lower = -20, upper = 20)
  
  return(1/(beta + 1)*int_term_f$value - 1/beta *cross_term$value + 1/(beta*(beta + 1))*int_term_g$value)
}


make_line_kld <- function(mu, sigma2, lambda) {
    KLD_G_minimiser <- optim(par = c(0, log(1)), fn = function(theta){KLD_univariate(log_pdf_g = function(x){log(normal_laplace_pdf2(x, mu, sqrt(sigma2), lambda))}, log_pdf_f = function(x){dnorm(x, theta[1], exp(theta[2]), log = TRUE)})})
    KLD_F0_KLD_G_minimiser <- KLD_univariate(log_pdf_g = function(x){dnorm(x, mu, sqrt(sigma2), log = TRUE)}, log_pdf_f = function(x){dnorm(x, KLD_G_minimiser$par[1], exp(KLD_G_minimiser$par[2]), log = TRUE)})
    return(KLD_F0_KLD_G_minimiser)
}

make_line_beta <- function(mu, sigma2, lambda, beta) {
    betaD_G_minimiser <- optim(par = c(0, log(1)), fn = function(theta){betaD_univariate(pdf_g = function(x){normal_laplace_pdf2(x, mu, sqrt(sigma2), lambda)}, pdf_f = function(x){dnorm(x, theta[1], exp(theta[2]))}, beta)})
    KLD_F0_betaD_G_minimiser <- KLD_univariate(log_pdf_g = function(x){dnorm(x, mu, sqrt(sigma2), log = TRUE)}, log_pdf_f = function(x){dnorm(x, betaD_G_minimiser$par[1], exp(betaD_G_minimiser$par[2]), log = TRUE)})
    return(KLD_F0_betaD_G_minimiser)
}




# BRANCHING PLOTS

branching_df <- data_without_reals %>%
    group_by(noise, real_n, synth_n, model_full) %>%
    summarise(
        exp_ll = mean(ll), exp_wass = mean(wass), exp_kld = mean(kld),
        ns = n(), se_ll = sd(ll) / sqrt(n()), se_wass = sd(wass) / sqrt(n()), se_kld = sd(kld) / sqrt(n()), 
        .groups="drop"
    ) %>%
    mutate(
        ci_ll = qt(conf_interval / 2 + .5, ns - 1) * se_ll,
        ci_wass = qt(conf_interval / 2 + .5, ns - 1) * se_wass,
        ci_kld = qt(conf_interval / 2 + .5, ns - 1) * se_kld
    )

unique_branches <- distinct(select(branching_df, c(noise, model_full)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            noise == vars[["noise"]],
            model_full == vars[["model_full"]]
        )
    plot_df_r <- branching_df %>%
        filter(
            synth_n == 0,
            noise == vars[["noise"]],
            model_full == vars[["model_full"]]
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        plot_df_r <- plot_df_r %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        p <- ggplot() +
            geom_errorbar(data=plot_df_r, aes(x=real_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
            geom_errorbar(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(real_n)), alpha=0.5) +
            geom_line(data=plot_df_r, aes(x=real_n, y=exp_metric)) +
            geom_line(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, color=as.factor(real_n))) +
            theme_light() +
            theme(legend.text=element_text(size=6)) + theme(legend.key.height = unit(5, "mm")) +
            xlim(c(0, 100)) +
            # scale_y_log10() +
            labs(x="Total Number of Samples (Real + Synth)", y=format_metric(metric), color="Number of\nReal Samples")
        
        if ((metric == "kld") & (vars[["model_full"]] %in% c("w = 0.5", "w = 1 (Naive)"))) {
            p <- p + geom_hline(yintercept=make_line_kld(0, 1, vars[["noise"]]), linetype="dashed")
        } else if ((metric == "kld") & (!vars[["model_full"]] %in% c("Noise-Aware", "Resampled", "w = 0 (No Syn)"))) {
            p <- p + geom_hline(yintercept=make_line_beta(0, 1, vars[["noise"]], as.numeric(tail(strsplit(vars[["model_full"]], " ")[[1]], n=1))), linetype="dashed")
        }

        ggsave(paste0(out_path, "/branched___", metric, "__noise_", vars[["noise"]], "__model_full_", vars[["model_full"]], ".pdf"), p, dpi=320, height=4, width=7, device=cairo_pdf)
    }

}


unique_branches <- distinct(select(branching_df, c(noise)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            noise == vars[["noise"]],
        ) %>% rowwise() %>% mutate(
        hline_val = ifelse(
            model_full %in% c("w = 0.5", "w = 1 (Naive)"), 
            make_line_kld(0, 1, vars[["noise"]]),
            ifelse(
                model_full %in% c("Resampled", "Noise-Aware", "w = 0.5", "w = 1 (Naive)", "w = 0 (No Syn)"),
                NA,
                make_line_beta(0, 1, vars[["noise"]], as.numeric(tail(strsplit(model_full, " ")[[1]], n=1)))
            )
        )
    )
    plot_df_r <- branching_df %>%
        filter(
            synth_n == 0,
            noise == vars[["noise"]],
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        plot_df_r <- plot_df_r %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )

        p <- ggplot() +
            facet_wrap(. ~ model_full) +
            geom_errorbar(data=plot_df_r, aes(x=real_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
            geom_errorbar(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(real_n)), alpha=0.5) +
            geom_line(data=plot_df_r, aes(x=real_n, y=exp_metric)) +
            geom_line(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, color=as.factor(real_n))) +
            theme_light() +
            # theme(legend.text=element_text(size=6)) + theme(legend.key.height = unit(5, "mm")) +
            xlim(c(0, 100)) +
            # scale_y_log10() +
            labs(x="Total Number of Samples (Real + Synth)", y=format_metric(metric), color="Number of\nReal Samples")
        if (metric == "kld") {
            p <- p + geom_hline(data=plot_df, aes(yintercept=hline_val), linetype="dashed")
        }
        ggsave(paste0(out_path, "/branched_all_models__", metric, "__noise_", vars[["noise"]], ".pdf"), p, dpi=320, height=8, width=14, device=cairo_pdf)
    }

}

#####

plot_df <- branching_df %>% filter(noise == 0.75, model_full %in% c("w = 0.5", paste0(intToUtf8(0x03B2), " = 0.5"))) %>%
    mutate(hline_val = ifelse(model_full == "w = 0.5", make_line_kld(0, 1, 0.75), make_line_beta(0, 1, 0.75, 0.5)))
plot_df_r <- branching_df %>% filter(synth_n == 0, model_full %in% c("w = 0.5", paste0(intToUtf8(0x03B2), " = 0.5")))
metric <- "kld"
plot_df <- plot_df %>% mutate(
    exp_metric = !!sym(paste0("exp_", metric)),
    ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
    ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
)
plot_df_r <- plot_df_r %>% mutate(
    exp_metric = !!sym(paste0("exp_", metric)),
    ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
    ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
)
p <- ggplot() +
    facet_wrap(. ~ model_full) +
    geom_errorbar(data=plot_df_r, aes(x=real_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
    geom_errorbar(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(real_n)), alpha=0.5) +
    geom_line(data=plot_df_r, aes(x=real_n, y=exp_metric)) +
    geom_line(data=plot_df, aes(x=real_n + synth_n, y=exp_metric, color=as.factor(real_n))) +
    geom_hline(data=plot_df, aes(yintercept=hline_val), linetype="dashed") +
    theme_light() +
    theme(legend.text=element_text(size=6)) + theme(legend.key.height = unit(5, "mm")) +
    xlim(c(0, 100)) +
    # scale_y_log10() +
    labs(x="Total Number of Samples (Real + Synth)", y=format_metric(metric), color="Number of\nReal Samples")
ggsave(paste0(out_path, "/joined_branched___", metric, ".pdf"), p, dpi=320, height=4, width=14, device=cairo_pdf)

#####

unique_branches <- distinct(select(branching_df, c(noise, real_n)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            real_n == vars[["real_n"]],
            noise == vars[["noise"]]
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        p <- ggplot() +
            # facet_wrap(. ~ real_n) +
            geom_errorbar(data=plot_df, aes(x=synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
            geom_line(data=plot_df, aes(x=synth_n, y=exp_metric, color=as.factor(model_full))) +
            labs(x="Number of Synthetic Samples", y=format_metric(metric), color="Model\nConfiguration") +
            scale_color_manual(values=c(noiseaware, w0, w05, w1, b05, b075)) +
            theme_light()
        ggsave(paste0(out_path, "/branched_fix_real___", metric, "__noise_", vars[["noise"]], "__real_n_", vars[["real_n"]], ".pdf"), p, dpi=320, height=4, width=7, device=cairo_pdf)
    }

}

unique_branches <- distinct(select(branching_df, c(noise)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            noise == vars[["noise"]]
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        p <- ggplot() +
            facet_wrap(. ~ real_n) +
            geom_errorbar(data=plot_df, aes(x=synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
            geom_line(data=plot_df, aes(x=synth_n, y=exp_metric, color=as.factor(model_full))) +
            labs(x="Number of Synthetic Samples", y=format_metric(metric), color="Model\nConfiguration") +
            scale_color_manual(values=c(noiseaware, w0, w05, w1, b05, b075)) +
            theme_light()
        ggsave(paste0(out_path, "/branched_all_real___", metric, "__noise_", vars[["noise"]], ".pdf"), p, dpi=320, height=16, width=14, device=cairo_pdf)
    }

}










min_data <- data_without_reals %>%
    group_by(real_n, seed, model_full, noise) %>%
    arrange(synth_n) %>%
    summarise(
        min_ll = min(ll),
        min_kld = min(kld), 
        min_wass = min(wass), 
        min_synth_n_ll = synth_n[which.min(ll)], 
        min_synth_n_kld = synth_n[which.min(kld)], 
        min_synth_n_wass = synth_n[which.min(wass)], 
        .groups="drop"
    ) 
exp_min_data <- min_data %>%
    group_by(model_full, noise, real_n) %>%
    summarise(
        exp_min_metric_kld = mean(min_kld),
        exp_min_metric_ll = mean(min_ll),
        exp_min_metric_wass = mean(min_wass),
        exp_min_synth_n_ll = mean(min_synth_n_ll), 
        exp_min_synth_n_kld = mean(min_synth_n_kld), 
        exp_min_synth_n_wass = mean(min_synth_n_wass), 
        .groups="drop"
    )

exp_data <- data_without_reals %>%
    group_by(real_n, synth_n, model_full, noise) %>%
    summarise(
        exp_ll = mean(ll),
        exp_kld = mean(kld),
        exp_wass = mean(wass),
        .groups="drop"
    )
min_exp_data <- exp_data %>%
    group_by(real_n, model_full, noise) %>%
    arrange(synth_n) %>%
    summarise(
        min_exp_metric_ll = min(exp_ll),
        min_exp_metric_kld = min(exp_kld),
        min_exp_metric_wass = min(exp_wass),
        min_exp_synth_n_ll = synth_n[which.min(exp_ll)], 
        min_exp_synth_n_kld = synth_n[which.min(exp_kld)], 
        min_exp_synth_n_wass = synth_n[which.min(exp_wass)], 
        .groups="drop"
    )



# MIN EXP VS EXP MIN PLOTS

plotting_data <- inner_join(exp_min_data, min_exp_data)
unique_plotting_data <- distinct(select(plotting_data, c(model_full, noise)))
for (i in 1:nrow(unique_plotting_data)) {

    vals = unique_plotting_data[i,]

    for (metric in metrics) {

        vals = c(vals[[1]], vals[[2]], metric)
        pd <- plotting_data %>% 
            filter(model_full==vals[1], noise==vals[2]) %>%
            select(real_n, contains(paste0("metric_", vals[3]))) %>%
            gather(order, metric, 2:3)
        p <- ggplot(pd, aes(x=real_n, y=metric)) + 
            geom_line(aes(color=order)) +
            labs(x="Number of Real Samples", y=format_metric(vals[3])) +
            ggtitle(as.character(vals)) +
            theme_light()
        ggsave(paste0(out_path, "/exp_min__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".pdf"), p, device=cairo_pdf)
        
        vals = c(vals[[1]], vals[[2]], metric)
        pd <- plotting_data %>% 
            filter(model_full==vals[1], noise==vals[2]) %>%
            select(real_n, contains(paste0("synth_n_", vals[3]))) %>%
            gather(order, metric, 2:3)
        p <- ggplot(pd, aes(x=real_n, y=metric)) + 
            geom_line(aes(color=order)) +
            labs(x="Number of Real Samples", y="Number of Synthetic Samples") +
            ggtitle(as.character(vals)) +
            theme_light()
        ggsave(paste0(out_path, "/exp_min_n__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".pdf"), p, device=cairo_pdf)

    }

}



# SPAGHETTI PLOTS

spag_data <- data_without_reals %>% 
    inner_join(min_data) %>% 
    inner_join(exp_data) %>% 
    inner_join(min_exp_data) %>% 
    inner_join(exp_min_data)

for (i in 1:nrow(distinct(select(spag_data, c(model_full, real_n, noise))))) {

    vals <- distinct(select(data, c(model_full, real_n, noise)))[i,]
    fd <- spag_data %>% filter(model_full == vals[[1]], real_n == vals[[2]], noise == vals[[3]])
    for (metric in metrics) {
    
        p <- ggplot(fd) +
            geom_line(aes_string(x="synth_n", y=metric, group="seed"), alpha=0.05) +
            geom_line(aes_string(x="synth_n", y=paste0("exp_", metric))) +
            geom_hline(aes_string(yintercept=paste0("exp_min_metric_", metric), color='"Expected Minimum"')) +
            geom_vline(aes_string(xintercept=paste0("exp_min_synth_n_", metric), color='"Expected Minimum"')) +
            geom_hline(aes_string(yintercept=paste0("min_exp_metric_", metric), color='"Minimum Expected"')) +
            geom_vline(aes_string(xintercept=paste0("min_exp_synth_n_", metric), color='"Minimum Expected"')) +
            theme_light()
            # geom_smooth(aes_string(x="synth_n", y=metric))
        ggsave(paste0(out_path, "/spag__model_", vals[[1]], "__real_n_", vals[[2]], "__noise_", vals[[3]], "__metric_", metric, ".pdf"), p, device=cairo_pdf)
    
    }

}











# ALPHA PLOTS

mean_df <- data %>%
    group_by(noise, model, alpha, weight, real_n, synth_n) %>%
    summarise(
        ll = mean(ll),
        kld = mean(kld),
        wass = mean(wass),
        .groups="drop"
    )

plotting_df <- mean_df %>%
    group_by(noise, model, real_n, synth_n) %>%
    summarise(
        ll_val = min(ll),
        kld_val = min(kld),
        wass_val = min(wass),
        ll = alpha[which.min(ll)],
        kld = alpha[which.min(ll)],
        wass = alpha[which.min(ll)],
        .groups="drop"
    )


groups <- distinct(select(plotting_df, c(noise, model, real_n)))

for (i in 1:nrow(groups)) {
    row <- groups[i,]
    plot_df_1 <- plotting_df %>%
        filter(noise == row[["noise"]], model == row[["model"]], real_n == row[["real_n"]])
    plot_df_2 <- mean_df %>%
        mutate(synth_n = as.factor(synth_n)) %>%
        filter(noise == row[["noise"]], model == row[["model"]], real_n == row[["real_n"]])
    for (metric in metrics) {
        p <- ggplot(plot_df_1, aes_string(x="synth_n", y=metric)) +
            geom_smooth() +
            geom_jitter(aes_string(color=paste0(metric, "_val"))) + ylab("Alpha")
        ggsave(paste0(out_path, "/alpha__real_n_", row[["real_n"]], "__noise_", row[["noise"]], "__metric_", metric, ".pdf"), p, device=cairo_pdf)
        p <- ggplot(plot_df_2, aes_string(x="alpha", y=metric, colour="synth_n")) + geom_line()
        ggsave(paste0(out_path, "/cross_sectional__real_n_", row[["real_n"]], "__noise_", row[["noise"]], "__metric_", metric, ".pdf"), p, device=cairo_pdf)
    }
}









# EPSILON DEMONSTRATIONS PLOTS

grouped_data <- data %>% 
    group_by(epsilon, real_n, synth_n) %>%
    summarise(
        ll = mean(ll), kld = mean(kld), wass = mean(wass), 
        se_ll = mean(ll), se_kld = mean(kld), se_wass = mean(wass)
    ) %>%
    filter(epsilon %in% c(0.01, 0.1, 1, 6, 10, 100))

real_baseline_100 <- grouped_data %>% filter(real_n == 100, synth_n == 0)
real_baseline_10 <- grouped_data %>% filter(real_n == 5, synth_n == 0)
synth_total_100 <- grouped_data %>% filter(synth_n == 100, real_n == 0)
synth_total_10 <- grouped_data %>% filter(synth_n == 5, real_n == 0)
synth_best <- grouped_data %>%
    filter(real_n == 0, synth_n > 0) %>%
    group_by(epsilon) %>%
    summarise(ll = min(ll), kld = min(kld), wass = min(wass))

breaks <- 10^(-2:3)
minor_breaks <- rep(1:9, 6)*(10^rep(-2:3, each=9))

for (metric in metrics) {
    p <- ggplot() +
        geom_line(data = real_baseline_100, aes(x=epsilon, y=!!sym(metric), color="N = 100", linetype="Real")) +
        geom_line(data = real_baseline_10, aes(x=epsilon, y=!!sym(metric), color="N = 5", linetype="Real")) +
        geom_line(data = synth_total_100, aes(x=epsilon, y=!!sym(metric), color="N = 100", linetype="Synthetic")) +
        geom_line(data = synth_total_10, aes(x=epsilon, y=!!sym(metric), color="N = 5", linetype="Synthetic")) +
        geom_line(data = synth_best, aes(x=epsilon, y=!!sym(metric), color="N = Optimal", linetype="Synthetic")) +
        scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x), labels = trans_format("log10", math_format(10^.x))) +
        scale_y_log10(breaks = breaks, minor_breaks = minor_breaks) +
        labs(x = "\u03B5", y=format_metric(metric)) +
        theme_light() +
        annotation_logticks()
    ggsave(paste0(out_path, "/noise_demo_gaussian___metric_", metric, ".pdf"), p, width=7, height=4, dpi=320, device=cairo_pdf)
}













epsilons <- c("0.0001", "0.001", "0.01", "0.1", "1.0", "10.0", "100.0", "1000.0")
vals <- c()
real_vals <- c()
for (eps in epsilons) {
    print(eps)
    data <- import(paste0("data/splits/framingham_TenYearCHD_eps", eps, "_synth.csv"))
    vals <- c(vals, sqrt(sum(diag(cov(data))) / dim(data)[2]))
    real_data <- import(paste0("data/splits/framingham_TenYearCHD_eps", eps, "_real.csv"))
    real_vals <- c(real_vals, sqrt(sum(diag(cov(real_data))) / dim(real_data)[2]))
}

plotty_df <- tibble(epsilon = 10^(-4:3), val = vals, real_val = real_vals)
p <- ggplot(plotty_df) + 
    geom_line(aes(x=epsilon, y=val, linetype="Synthetic")) +
    geom_hline(aes(yintercept=real_val, linetype="Real")) +
    theme_light() +
    labs(x = "\u03B5", y="Average Standard Deviation per Predictor", linetype="Dataset Type") +
    scale_x_log10() + scale_y_log10()

ggsave(paste0("from_cluster/logistic_regression/plots/gan_std_comparison.pdf"), p, width=7, height=4, dpi=320, device=cairo_pdf)

