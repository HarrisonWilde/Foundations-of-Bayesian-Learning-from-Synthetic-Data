library(rio)
library(ggplot2)
library(dplyr)
library(tidyverse)


format_metric <- function(str) {
    if (str == "kld") {
        return("Kullback-Leibler Distance")
    } else if (str == "wass") {
        return("Wasserstein Distance")
    } else if (str == "ll") {
        return("Log Score")
    } else {
        return("No Formatting Given for this Metric")
    }
}


path <- "from_cluster/gaussian/outputs/final_csvs/neff.csv"
# path <- "gaussian/outputs/noise_demo.csv"
metrics <- c("kld", "ll", "wass")
out_path <- str_remove(str_replace(paste(str_split(path, "/")[[1]][-4], collapse="/"), "outputs", "plots"), ".csv")
dir.create(out_path)

data <- drop_na(import(path, setclass="tibble", fill=TRUE), iter)






data <- data %>%
    mutate(model_full = ifelse(model == "noise_aware", model, ifelse(model == "beta", paste0(model, "_", weight, "_", beta), paste0(model, "_", weight)))) %>% 
    select(-c(model, beta, weight))

real_ns = c(2, 4, 6, 8, 10, 13, 16, 19, 22, 25, 30, 35, 40, 50, 75, 100)

data_without_reals <- data %>% filter(real_n %in% real_ns)







# unique_branches <- distinct(select(data_without_reals, c(noise, model_full)))

# for (i in 1:nrow(unique_branches)) {
#     ub <- unique_branches[i,]
#     fdf_wr <- data_without_reals %>%
#         filter(noise == ub[["noise"]], model_full == ub[["model_full"]])
#     fdf <- data %>%
#         filter(noise == ub[["noise"]], model_full == ub[["model_full"]], synth_n == 0)
#     for (metric in metrics) {
#         metric_sym <- sym(metric)
#         p <- ggplot() +
#             geom_smooth(data=fdf_wr, aes(x=real_n + synth_n, color=as.factor(real_n), y=!!metric_sym)) +
#             geom_smooth(data=fdf, aes(x=real_n, y=!!metric_sym)) +
#             labs(x="Total Samples", y=format_metric(metric))
#         ggsave(paste0(out_path, "/branched___", metric, "__noise_", ub[["noise"]], "__model_full_", ub[["model_full"]], ".png"), p)
#     }
# }

conf_interval <- 0.95

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
            # scale_y_log10() +
            labs(x="Total Samples (Real N + Synth N)", y=format_metric(metric), color="Number of\nReal Samples")
        ggsave(paste0(out_path, "/branched___", metric, "__noise_", vars[["noise"]], "__model_full_", vars[["model_full"]], ".png"), p, dpi=320, height=8, width=11)
    }

}

unique_branches <- distinct(select(branching_df, c(noise, real_n)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            noise == vars[["noise"]],
            real_n == vars[["real_n"]]
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("se_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("se_", metric))),
        )
        p <- ggplot() +
            geom_errorbar(data=plot_df, aes(x=synth_n, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
            geom_line(data=plot_df, aes(x=synth_n, y=exp_metric, color=as.factor(model_full))) +
            labs(x="Number of Synthetic Samples", y=format_metric(metric))
        ggsave(paste0(out_path, "/branched_fix_real___", metric, "__noise_", vars[["noise"]], "__real_n_", vars[["real_n"]], ".png"), p, dpi=320, height=8, width=11)
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
            ggtitle(as.character(vals))
        ggsave(paste0(out_path, "/exp_min__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".png"), p)
        
        vals = c(vals[[1]], vals[[2]], metric)
        pd <- plotting_data %>% 
            filter(model_full==vals[1], noise==vals[2]) %>%
            select(real_n, contains(paste0("synth_n_", vals[3]))) %>%
            gather(order, metric, 2:3)
        p <- ggplot(pd, aes(x=real_n, y=metric)) + 
            geom_line(aes(color=order)) +
            labs(x="Number of Real Samples", y="Number of Synthetic Samples") +
            ggtitle(as.character(vals))
        ggsave(paste0(out_path, "/exp_min_n__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".png"), p)

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
            geom_vline(aes_string(xintercept=paste0("min_exp_synth_n_", metric), color='"Minimum Expected"'))
            # geom_smooth(aes_string(x="synth_n", y=metric))
        ggsave(paste0(out_path, "/spag__model_", vals[[1]], "__real_n_", vals[[2]], "__noise_", vals[[3]], "__metric_", metric, ".png"), p)
    
    }

}



# Bootstrap NEFF plots

n <- 100
B <- 50



# ADDITIONAL N PLOTS

real_avgs <- data %>% 
    filter(synth_n == 0) %>% 
    group_by(real_n) %>% 
    summarise(real_ll = mean(ll), real_kld = mean(kld), real_wass = mean(wass))

new_cols <- matrix(NA, nrow=nrow(min_exp_data), ncol=length(metrics))
for (i in 1:nrow(min_exp_data)) {
    row = min_exp_data[i,]
    add_row = c()
    for (metric in metrics) {
        val <- row[[paste0("min_exp_", metric)]]
        ordered <- real_avgs %>% arrange(abs(!!sym(paste0("real_", metric)) - val)) %>% select(real_n)
        add_row = c(add_row, ordered[1,][[1]])
    }
    new_cols[i,] = add_row
}

n_add_col_names <- c()
for (metric in metrics) {
    n_add_col_names <- c(n_add_col_names, paste0("n_add_", metric))
}
colnames(new_cols) <- n_add_col_names
n_add_data <- cbind(min_exp_data, new_cols)



# FIX THIS

n_add_data <- n_add_data %>% mutate(n_eff_kld = n_add_kld - real_n, n_eff_ll = n_add_ll - real_n, n_eff_wass = n_add_wass - real_n)

# for (metric in metrics) {
#     n_add_data <- n_add_data %>% mutate(!!paste0("n_eff_", metric) = !!paste0("n_add_", metric) - real_n)
# }

for (i in 1:nrow(distinct(select(min_exp_data, noise)))) {
    noise <- distinct(select(min_exp_data, noise))[i,][[1]]
    for (metric in metrics) {
        p <- ggplot(n_add_data %>% filter(!!noise == noise), aes_string(x="real_n", y=paste0("n_eff_", metric), color="model_full")) + geom_line()
        ggsave(paste0(out_path, "/n_add__noise_", noise, "__metric_", metric, ".png"), p)
    }
}










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
        ggsave(paste0(out_path, "/alpha__real_n_", row[["real_n"]], "__noise_", row[["noise"]], "__metric_", metric, ".png"), p)
        p <- ggplot(plot_df_2, aes_string(x="alpha", y=metric, colour="synth_n")) + geom_line()
        ggsave(paste0(out_path, "/cross_sectional__real_n_", row[["real_n"]], "__noise_", row[["noise"]], "__metric_", metric, ".png"), p)
    }
}







# EPSILON DEMONSTRATIONS PLOTS

grouped_data <- data %>% 
    group_by(epsilon, real_n, synth_n) %>%
    summarise(ll = mean(ll), kld = mean(kld), wass = mean(wass))

real_baseline_100 <- grouped_data %>% filter(real_n == 100, synth_n == 0)
real_baseline_10 <- grouped_data %>% filter(real_n == 10, synth_n == 0)
synth_total_100 <- grouped_data %>% filter(synth_n == 100, real_n == 0)
synth_total_10 <- grouped_data %>% filter(synth_n == 10, real_n == 0)
synth_best <- grouped_data %>% filter(real_n == 0, synth_n > 0) %>% group_by(epsilon) %>% summarise(ll = min(ll), kld = min(kld), wass = min(wass))

breaks <- 10^(-2:3)
minor_breaks <- rep(1:9, 6)*(10^rep(-2:3, each=9))

ggplot() +
    geom_line(data = real_baseline_100, aes(x=epsilon, y=ll, color="N = 100", linetype="Real")) +
    geom_line(data = real_baseline_10, aes(x=epsilon, y=ll, color="N = 10", linetype="Real")) +
    geom_line(data = synth_total_100, aes(x=epsilon, y=ll, color="N = 100", linetype="Synthetic")) +
    geom_line(data = synth_total_10, aes(x=epsilon, y=ll, color="N = 10", linetype="Synthetic")) +
    geom_line(data = synth_best, aes(x=epsilon, y=ll, color="N = Optimal", linetype="Synthetic")) +
    scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
    scale_y_log10(breaks = breaks, minor_breaks = minor_breaks) +
    annotation_logticks()
