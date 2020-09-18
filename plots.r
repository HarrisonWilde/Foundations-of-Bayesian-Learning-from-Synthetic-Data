library(rio)
library(ggplot2)
library(dplyr)
library(tidyverse)


path <- "from_cluster/gaussian/outputs/final_csvs/granular2.csv"
metrics <- c("kld", "ll", "wass")
out_path <- str_remove(str_replace(paste(str_split(path, "/")[[1]][-4], collapse="/"), "outputs", "plots"), ".csv")
dir.create(out_path)

data <- drop_na(import(path, setclass="tibble", fill=TRUE), iter)
data <- data %>%
    mutate(model_full = ifelse(model == "noise_aware", model, ifelse(model == "beta", paste0(model, "_", weight, "_", beta), paste0(model, "_", weight)))) %>% 
    select(-c(model, beta, weight))

real_ns = c(5, 10, 15, 20, 25, 50, 100)

data_without_reals <- data %>% filter(real_n %in% real_ns)

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
        ordered <- real_avgs %>% arrange(abs(!!as.symbol(paste0("real_", metric)) - val)) %>% select(real_n)
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