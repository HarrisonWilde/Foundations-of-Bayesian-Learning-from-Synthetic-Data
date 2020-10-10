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
    } else if (str == "param_mse") {
        return("Total Parameter Posterior Squared Distance")
    } else if (str == "auc") {
        return("AUROC")
    } else {
        return("No Formatting Given for this Metric")
    }
}

gg_color_hue <- function(n) {
    hues = seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
}


path <- "from_cluster/logistic_regression/outputs/final_csvs/heart.csv"
# path <- "from_cluster/logistic_regression/outputs/final_csvs/grid_framingham_eps0p01.csv"
metrics <- c("auc", "ll", "param_mse")
out_path <- str_remove(str_replace(paste(str_split(path, "/")[[1]][-4], collapse="/"), "outputs", "plots"), ".csv")
dir.create(out_path)

data <- drop_na(import(path, setclass="tibble", fill=TRUE), iter)






data <- data %>%
    mutate(model_full = ifelse(model == "noise_aware", model, ifelse(model == "beta", paste0(model, "_", weight, "_", beta), paste0(model, "_", weight)))) %>% 
    select(-c(model, beta, weight))

# real_ns = c(2, 4, 6, 8, 10, 13, 16, 19, 22, 25, 30, 35, 40, 50, 75, 100)

# data_without_reals <- data %>% filter(real_n %in% real_ns)


conf_interval <- 0.95

branching_df <- data %>%
    group_by(real_alpha, synth_alpha, model_full) %>%
    summarise(
        exp_ll = mean(ll), exp_auc = mean(auc), exp_param_mse = mean(param_mse),
        ns = n(), se_ll = sd(ll) / sqrt(n()), se_auc = sd(auc) / sqrt(n()), se_param_mse = sd(param_mse) / sqrt(n()), 
        .groups="drop"
    ) %>%
    mutate(
        ci_ll = qt(conf_interval / 2 + .5, ns - 1) * se_ll,
        ci_auc = qt(conf_interval / 2 + .5, ns - 1) * se_auc,
        ci_param_mse = qt(conf_interval / 2 + .5, ns - 1) * se_param_mse
    )

unique_branches <- distinct(select(branching_df, c(model_full)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            model_full == vars[["model_full"]]
        ) %>%
        mutate(real_factor = factor(real_alpha))
    plot_df_r <- branching_df %>%
        filter(
            synth_alpha == 0,
            model_full == vars[["model_full"]]
        )
    
    # hues <- c(gg_color_hue(nrow(distinct(select(plot_df, real_alpha)))), "black")

    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))),
        )
        plot_df_r <- plot_df_r %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))),
        )
        p <- ggplot() +
            geom_errorbar(data=plot_df_r, aes(x=100 * real_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
            geom_errorbar(data=plot_df, aes(x=100 * (real_alpha + synth_alpha), y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=reorder(factor(real_alpha * 100), sort(real_alpha))), alpha=0.5) +
            geom_line(data=plot_df_r, aes(x=100 * real_alpha, y=exp_metric)) +
            geom_line(data=plot_df, aes(x=100 * (real_alpha + synth_alpha), y=exp_metric, color=reorder(factor(real_alpha * 100), sort(real_alpha)))) +
            labs(x="Total Dataset Utilisation (Real % + Synth %)", y=format_metric(metric), color="% of Real\nDataset Used") +
            # scale_color_manual(values=hues) +
            ggtitle(paste0("Real and Synthetic dataset utilisation mixes for ", vars[["model_full"]], " in terms of ", format_metric(metric)))

        ggsave(paste0(out_path, "/branched___", metric, "__model_full_", vars[["model_full"]], ".png"), p)
    }

}

unique_branches <- distinct(select(branching_df, c(real_alpha)))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(
            real_alpha == vars[["real_alpha"]]
        )
    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))),
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))),
        )
        p <- ggplot() +
            geom_errorbar(data=plot_df, aes(x=100 * synth_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
            geom_line(data=plot_df, aes(x=100 * synth_alpha, y=exp_metric, color=as.factor(model_full))) +
            labs(x="Synthetic Dataset % Used", y=format_metric(metric)) +
            ggtitle(paste0("Model config comparisons for ", format_metric(metric), " with ", vars[["real_alpha"]] * 100, "% real dataset utilisation"))
        ggsave(paste0(out_path, "/branched_fix_real___", metric, "__real_alpha_", vars[["real_alpha"]], ".png"), p)
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
            labs(x="Real Number of Samples", y=vals[3]) +
            ggtitle(as.character(vals))
        ggsave(paste0(out_path, "/exp_min__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".png"), p)
        
        vals = c(vals[[1]], vals[[2]], metric)
        pd <- plotting_data %>% 
            filter(model_full==vals[1], noise==vals[2]) %>%
            select(real_n, contains(paste0("synth_n_", vals[3]))) %>%
            gather(order, metric, 2:3)
        p <- ggplot(pd, aes(x=real_n, y=metric)) + 
            geom_line(aes(color=order)) +
            labs(x="Real Number of Samples", y=vals[3]) +
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

