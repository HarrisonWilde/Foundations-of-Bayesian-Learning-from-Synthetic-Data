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

dataset <- "framingham"
# path <- paste0("logistic_regression/outputs/noise_demo.csv")
# path <- paste0("from_cluster/logistic_regression/outputs/final_csvs/", dataset, ".csv")
path <- "from_cluster/logistic_regression/outputs/final_csvs/framingham_eps0p0001.csv"
metrics <- c("auc", "ll", "param_mse")
out_path <- str_remove(str_replace(paste(str_split(path, "/")[[1]][-4], collapse="/"), "outputs", "plots"), ".csv")
dir.create(out_path)

data <- drop_na(import(path, setclass="tibble", fill=TRUE), auc)
# data <- bind_rows(data, drop_na(import(path, setclass="tibble", fill=TRUE), auc))

data <- data %>% filter(weight %in% c(0, 1, 0.5, 1.25))

if (dataset == "framingham") {
    multiplier <- 4240
} else if (dataset == "heart") {
    multiplier <- 303
}
# multiplier <- 100
# data <- data %>% mutate(real_n = round(real_alpha * multiplier), synth_n = round(synth_alpha * multiplier))



data <- data %>%
    mutate(model_full = ifelse(
        model == "noise_aware", "Noise-Aware", ifelse(
            model == "beta", paste0(intToUtf8(0x03B2), " = ", beta), ifelse(
                model == "resampled", "Resampled", ifelse(
                    ((model == "weighted") & (weight == 0)), "w = 0 (No Syn)", ifelse(
                        ((model == "weighted") & (weight == 1)), "w = 1 (Naive)", paste0("w = ", weight))))))) %>% 
    select(-c(model, beta, weight))

real_alphas = c(0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0)

data_without_reals <- data %>% filter(real_alpha %in% real_alphas)


conf_interval <- 0.95

branching_df <- data_without_reals %>%
    group_by(epsilon, real_alpha, synth_alpha, model_full) %>%
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

branching_df_r <- data %>%
    filter(synth_alpha == 0) %>%
    group_by(real_alpha, synth_alpha) %>%
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

# branching_df <- branching_df %>% filter(!epsilon %in% c(0.0001, 6)) %>% mutate(epsilon = replace(epsilon, epsilon == 1000, 100))

for (i in 1:nrow(unique_branches)) {

    vars <- unique_branches[i,]
    plot_df <- branching_df %>%
        filter(model_full == vars[["model_full"]]) %>%
        mutate(real_factor = factor(paste0(round(real_alpha * multiplier), " (", real_alpha * 100, "%)"))) %>% filter(real_alpha >= 0.05)
    plot_df_r <- branching_df_r %>% filter(real_alpha >= 0.05)
    
    # hues <- c(gg_color_hue(nrow(distinct(select(plot_df, real_alpha)))), "black")

    for (metric in metrics) {
        plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
        )
        plot_df_r <- plot_df_r %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
        )
        p <- ggplot() +
            geom_errorbar(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
            geom_errorbar(data=plot_df, aes(x=multiplier * (real_alpha + synth_alpha), y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=reorder(real_factor, sort(real_alpha))), alpha=0.5) +
            geom_line(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric)) +
            geom_line(data=plot_df, aes(linetype=factor(epsilon), x=multiplier * (real_alpha + synth_alpha), y=exp_metric, color=reorder(real_factor, sort(real_alpha)))) +
            labs(x="Total Number of Samples (Real + Synth)", color="Number of\nReal Samples", linetype="\u03B5") +
            # ggtitle(paste0("Real and Synthetic dataset utilisation mixes for ", vars[["model_full"]], " in terms of ", format_metric(metric))) +
            theme_light()
            # scale_color_manual(values=hues) +
        
        
        if ((metric == "param_mse") || (metric == "ll")) {
            p <- p +
                scale_y_log10() +
                annotation_logticks(sides = "l") +
                ylab(paste0("Log(", format_metric(metric), ")"))
        } else {
            p <- p + ylab(paste0(format_metric(metric)))
        }
        
        ggsave(paste0(out_path, "/many_eps_branched___", metric, "__model_full_", vars[["model_full"]], ".png"), p, height=4, dpi=320)
    }

}


plot_df <- branching_df %>%
    mutate(real_factor = factor(paste0(round(real_alpha * multiplier), " (", real_alpha * 100, "%)"))) %>% filter(real_alpha >= 0.05)
plot_df_r <- branching_df_r %>% filter(real_alpha >= 0.05)

# hues <- c(gg_color_hue(nrow(distinct(select(plot_df, real_alpha)))), "black")

for (metric in metrics) {
    plot_df <- plot_df %>% mutate(
        exp_metric = !!sym(paste0("exp_", metric)),
        ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
        ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
    )
    plot_df_r <- plot_df_r %>% mutate(
        exp_metric = !!sym(paste0("exp_", metric)),
        ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
        ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
    )
    p <- ggplot() +
        facet_wrap(. ~ model_full) +
        geom_errorbar(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
        geom_errorbar(data=plot_df, aes(x=multiplier * (real_alpha + synth_alpha), y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=reorder(real_factor, sort(real_alpha))), alpha=0.5) +
        geom_line(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric)) +
        geom_line(data=plot_df, aes(x=multiplier * (real_alpha + synth_alpha), y=exp_metric, color=reorder(real_factor, sort(real_alpha)))) +
        labs(x="Total Number of Samples (Real + Synth)", color="Number of\nReal Samples", linetype="\u03B5") +
        # ggtitle(paste0("Real and Synthetic dataset utilisation mixes for ", vars[["model_full"]], " in terms of ", format_metric(metric))) +
        theme_light()
        # scale_color_manual(values=hues) +
    
    
    if ((metric == "param_mse") || (metric == "ll")) {
        p <- p +
            scale_y_log10() +
            annotation_logticks(sides = "l") +
            ylab(paste0("Log(", format_metric(metric), ")"))
    } else {
        p <- p + ylab(paste0(format_metric(metric)))
    }
    
    ggsave(paste0(out_path, "/branched_all_models___", metric, ".png"), p, height=16, width=14, dpi=320)
}

####

plot_df <- branching_df %>%
    filter(model_full %in% c("w = 0.5", "w = 1 (Naive)", "β = 0.5, w = 1.25")) %>%
    mutate(real_factor = factor(paste0(round(real_alpha * multiplier), " (", real_alpha * 100, "%)"))) %>% filter(real_alpha >= 0.05)
plot_df_r <- branching_df_r %>% filter(real_alpha >= 0.05)

metric <- "auc"

plot_df <- plot_df %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
        )
        plot_df_r <- plot_df_r %>% mutate(
            exp_metric = !!sym(paste0("exp_", metric)),
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
        )
        p <- ggplot() +
        facet_wrap(. ~ model_full) +
            geom_errorbar(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric), alpha=0.5) +
            geom_errorbar(data=plot_df, aes(x=multiplier * (real_alpha + synth_alpha), y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=reorder(real_factor, sort(real_alpha))), alpha=0.5) +
            geom_line(data=plot_df_r, aes(x=multiplier * real_alpha, y=exp_metric)) +
            geom_line(data=plot_df, aes(x=multiplier * (real_alpha + synth_alpha), y=exp_metric, color=reorder(real_factor, sort(real_alpha)))) +
            labs(x="Total Number of Samples (Real n + Synth m)", color="Number of\nReal Samples") +
            # ggtitle(paste0("Real and Synthetic dataset utilisation mixes for ", vars[["model_full"]], " in terms of ", format_metric(metric))) +
            theme_light()
            # scale_color_manual(values=hues) +
        
        
if ((metric == "param_mse") || (metric == "ll")) {
    p <- p +
        scale_y_log10() +
        annotation_logticks(sides = "l") +
        ylab(paste0("Log(", format_metric(metric), ")"))
} else {
    p <- p + ylab(paste0(format_metric(metric)))
}

ggsave(paste0(out_path, "/branched___", metric, "__models_together.png"), p, height=4, width=14, dpi=320)

####

resampled <- "#ffc107"
w0 <- "#ffab91"
w05 <- "#ff5722"
w1 <- "#d84315"
b025 <- "#90caf9"
b05 <- "#2196f3"
b075 <- "#1565c0"

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
            ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
            ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
        )

        p <- ggplot() +
            geom_errorbar(data=plot_df, aes(x=multiplier * synth_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
            geom_line(data=plot_df, aes(x=multiplier * synth_alpha, y=exp_metric, color=as.factor(model_full))) +
            labs(x="Number of Synthetic Samples", color="Model\nConfiguration") +
            scale_color_manual(values=c(resampled, w0, w05, w1, b025, b05, b075)) +
            # ggtitle(paste0("Model config comparisons for ", format_metric(metric), " with ", vars[["real_alpha"]] * 100, "% real dataset utilisation")) +
            theme_light()
        
        if ((metric == "param_mse") || (metric == "ass")) {
            p <- p +
                scale_y_log10() +
                annotation_logticks(sides = "l") +
                ylab(paste0("Log(", format_metric(metric), ")"))
        } else {
            p <- p + ylab(paste0(format_metric(metric)))
        }
        
        ggsave(paste0(out_path, "/branched_fix_real___", metric, "__real_alpha_", vars[["real_alpha"]], ".png"), p, height=4, dpi=320)
    }

}

####
plot_df <- branching_df %>%
    mutate(real_factor = factor(paste0(round(real_alpha * multiplier), " (", real_alpha * 100, "%)"))) %>% filter(real_alpha >= 0.05)

for (metric in metrics) {

    plot_df <- plot_df %>% mutate(
        exp_metric = !!sym(paste0("exp_", metric)),
        ci_low_metric = !!sym(paste0("exp_", metric)) - !!(sym(paste0("ci_", metric))) / 2,
        ci_high_metric = !!sym(paste0("exp_", metric)) + !!(sym(paste0("ci_", metric))) / 2,
    )

    p <- ggplot() +
        facet_wrap(. ~ reorder(real_factor, sort(real_alpha))) +
        geom_errorbar(data=plot_df, aes(x=multiplier * synth_alpha, y=exp_metric, ymin=ci_low_metric, ymax=ci_high_metric, color=as.factor(model_full)), alpha=0.5) +
        geom_line(data=plot_df, aes(x=multiplier * synth_alpha, y=exp_metric, color=as.factor(model_full))) +
        labs(x="Number of Synthetic Samples", color="Model\nConfiguration") +
        scale_color_manual(values=c(resampled, w0, w05, w1, b025, b05, b075)) +
        # ggtitle(paste0("Model config comparisons for ", format_metric(metric), " with ", vars[["real_alpha"]] * 100, "% real dataset utilisation")) +
        theme_light()
    
    if ((metric == "param_mse") || (metric == "ass")) {
        p <- p +
            scale_y_log10() +
            annotation_logticks(sides = "l") +
            ylab(paste0("Log(", format_metric(metric), ")"))
    } else {
        p <- p + ylab(paste0(format_metric(metric)))
    }
    
    ggsave(paste0(out_path, "/branched_all_real___", metric, ".png"), p, height=16, width=14, dpi=320)
}

####





# N EFFECTIVE PLOTS

real_avgs <- data %>% 
    filter(synth_alpha == 0) %>% 
    group_by(real_alpha) %>% 
    summarise(
        matched_ll = mean(ll), matched_auc = mean(auc), matched_param_mse = mean(param_mse)
    ) %>%
    mutate(alpha_eff_ll = real_alpha, alpha_eff_auc = real_alpha, alpha_eff_param_mse = real_alpha) %>%
    select(-real_alpha)





# Bootstrap n effective plots

N <- 100
B <- 500

uniques <- data_without_reals %>% select(real_alpha) %>% distinct()

disty <- function(x, y) abs(x - y)

data_list <- list()

for (j in 1:nrow(uniques)) {
    u <- uniques[j,]
    n1 <- u[[1]]
    working_df <- data_without_reals %>% filter(real_alpha == n1) %>% arrange(model_full)
    seeds <- unique(working_df$seed)
    b_list <- list()
    for (i in 1:B) {
        b_seeds <- tibble(seed=sample(seeds, N, replace=TRUE))
        b_df <- b_seeds %>%
            inner_join(working_df) %>%
            group_by(synth_alpha, model_full) %>%
            summarise(b_exp_ll = mean(ll), b_exp_auc = mean(auc), b_exp_param_mse = mean(param_mse), .groups="drop") %>%
            group_by(model_full) %>%
            summarise(
                b_min_exp_ll = min(b_exp_ll), b_min_exp_auc = max(b_exp_auc), b_min_exp_param_mse = min(b_exp_param_mse),
                n_min_exp_ll = synth_alpha[which.min(b_exp_ll)], n_min_exp_auc = synth_alpha[which.max(b_exp_auc)], n_min_exp_param_mse = synth_alpha[which.min(b_exp_param_mse)], .groups="drop"
            )

        idx <- sapply(b_df$b_min_exp_ll, function(x) which.min( disty(x, real_avgs$matched_ll)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_ll, alpha_eff_ll))[idx,1:2,drop=FALSE])
        idx <- sapply(b_df$b_min_exp_auc, function(x) which.min( disty(x, real_avgs$matched_auc)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_auc, alpha_eff_auc))[idx,1:2,drop=FALSE])
        idx <- sapply(b_df$b_min_exp_param_mse, function(x) which.min( disty(x, real_avgs$matched_param_mse)))
        b_df <- bind_cols(b_df, select(real_avgs, c(matched_param_mse, alpha_eff_param_mse))[idx,1:2,drop=FALSE])
        b_df <- b_df %>% mutate(
            alpha_eff_ll = alpha_eff_ll - n1,
            alpha_eff_auc = alpha_eff_auc - n1,
            alpha_eff_param_mse = alpha_eff_param_mse - n1
        )
        b_list[[i]] <- b_df
    }
    working_df <- do.call(rbind, b_list)
    working_df <- working_df %>%
        group_by(model_full) %>%
        summarise(
            exp_alpha_eff_ll = mean(alpha_eff_ll),
            exp_alpha_eff_auc = mean(alpha_eff_auc),
            exp_alpha_eff_param_mse = mean(alpha_eff_param_mse),
            se_alpha_eff_ll = sd(alpha_eff_ll),
            se_alpha_eff_auc = sd(alpha_eff_auc),
            se_alpha_eff_param_mse = sd(alpha_eff_param_mse)
        )
    working_df <- bind_cols(
        tibble(real_alpha = rep(n1, nrow(working_df))),
        working_df
    )
    data_list[[j]] <- working_df
}

alpha_eff_plot_df <- do.call(rbind, data_list)
alpha_eff_plot_df <- alpha_eff_plot_df %>% filter(!model_full %in% c("Resampled", "w = 0 (No Syn)"))

for (metric in metrics) {
    alpha_eff_plot_df <- alpha_eff_plot_df %>% mutate(
        exp_metric = pmax(!!sym(paste0("exp_alpha_eff_", metric)), 0)
    ) %>% mutate(
        ci_low_metric = pmax(!!sym(paste0("exp_alpha_eff_", metric)), 0) - !!(sym(paste0("se_alpha_eff_", metric))) / 1.5,
        ci_high_metric = pmax(!!sym(paste0("exp_alpha_eff_", metric)), 0) + !!(sym(paste0("se_alpha_eff_", metric))) / 1.5,
    )
    p <- ggplot(
        alpha_eff_plot_df,
        aes(x=multiplier * real_alpha, y=multiplier * exp_metric, color=model_full)
    ) +
    geom_errorbar(
        aes(ymin=multiplier * ci_low_metric, ymax=multiplier * ci_high_metric),
        alpha = 0.5
    ) +
    geom_line() +
    labs(x = "Number of Real Samples", y = "Maximum Effective Real Sample Gain\nThroough the Use of Synthetic Data", color = "Model Type") +
    scale_color_manual(values=c(w05, w1, b025, b05, b075)) +
    theme_light()
    ggsave(paste0(out_path, "/alpha_eff___metric_", metric, ".png"), p, height=4, dpi=320)
}


part_1 <- alpha_eff_plot_df %>% mutate(
        exp_metric = pmax(!!sym(paste0("exp_alpha_eff_ll")), 0)
    ) %>% mutate(
        ci_low_metric = pmax(!!sym(paste0("exp_alpha_eff_ll")), 0) - !!(sym(paste0("se_alpha_eff_ll"))) / 1.5,
        ci_high_metric = pmax(!!sym(paste0("exp_alpha_eff_ll")), 0) + !!(sym(paste0("se_alpha_eff_ll"))) / 1.5,
    )
part_1 <- bind_cols(part_1, tibble(metric_name = rep("Log Score", nrow(part_1))))
part_2 <- alpha_eff_plot_df %>% mutate(
        exp_metric = pmax(!!sym(paste0("exp_alpha_eff_auc")), 0)
    ) %>% mutate(
        ci_low_metric = pmax(!!sym(paste0("exp_alpha_eff_auc")), 0) - !!(sym(paste0("se_alpha_eff_auc"))) / 2,
        ci_high_metric = pmax(!!sym(paste0("exp_alpha_eff_auc")), 0) + !!(sym(paste0("se_alpha_eff_auc"))) / 2,
    )
part_2 <- bind_cols(part_2, tibble(metric_name = rep("AUROC", nrow(part_2))))
both_parts <- rbind(part_1, part_2)

p <- ggplot(
        both_parts,
        aes(x=multiplier * real_alpha, y=multiplier * exp_metric, color=model_full)
    ) + facet_wrap(. ~ metric_name, scales = "free_y") +
    geom_errorbar(
        aes(ymin=multiplier * ci_low_metric, ymax=multiplier * ci_high_metric),
        alpha = 0.5
    ) +
    geom_line() +
    labs(x = "Number of Real Samples", y = "Maximum Effective Real Sample Gain\nThroough the Use of Synthetic Data", color = "Model Type") +
    scale_color_manual(values=c(w05, w1, b025, b05, b075)) +
    theme_light()
ggsave(paste0(out_path, "/alpha_eff___joined.png"), p, height=4, width=14, dpi=320)


min_exp_vs_exp_min(data_without_reals)

min_exp_vs_exp_min <- function(df) {

    min_data <- df %>%
        group_by(real_alpha, seed, model_full, noise) %>%
        arrange(synth_alpha) %>%
        summarise(
            min_ll = min(ll),
            min_auc = min(auc), 
            min_param_mse = min(param_mse), 
            min_synth_alpha_ll = synth_alpha[which.min(ll)], 
            min_synth_alpha_auc = synth_alpha[which.min(auc)], 
            min_synth_alpha_param_mse = synth_alpha[which.min(param_mse)], 
            .groups="drop"
        ) 
    exp_min_data <- min_data %>%
        group_by(model_full, noise, real_alpha) %>%
        summarise(
            exp_min_metric_auc = mean(min_auc),
            exp_min_metric_ll = mean(min_ll),
            exp_min_metric_param_mse = mean(min_param_mse),
            exp_min_synth_alpha_ll = mean(min_synth_alpha_ll), 
            exp_min_synth_alpha_auc = mean(min_synth_alpha_auc), 
            exp_min_synth_alpha_param_mse = mean(min_synth_alpha_param_mse), 
            .groups="drop"
        )

    exp_data <- df %>%
        group_by(real_alpha, synth_alpha, model_full, noise) %>%
        summarise(
            exp_ll = mean(ll),
            exp_auc = mean(auc),
            exp_param_mse = mean(param_mse),
            .groups="drop"
        )
    min_exp_data <- exp_data %>%
        group_by(real_alpha, model_full, noise) %>%
        arrange(synth_alpha) %>%
        summarise(
            min_exp_metric_ll = min(exp_ll),
            min_exp_metric_auc = min(exp_auc),
            min_exp_metric_param_mse = min(exp_param_mse),
            min_exp_synth_alpha_ll = synth_alpha[which.min(exp_ll)], 
            min_exp_synth_alpha_auc = synth_alpha[which.min(exp_auc)], 
            min_exp_synth_alpha_param_mse = synth_alpha[which.min(exp_param_mse)], 
            .groups="drop"
        )

    plotting_data <- inner_join(exp_min_data, min_exp_data)
    unique_plotting_data <- distinct(select(plotting_data, c(model_full, noise)))
    for (i in 1:nrow(unique_plotting_data)) {

        vals = unique_plotting_data[i,]

        for (metric in metrics) {

            vals = c(vals[[1]], vals[[2]], metric)
            pd <- plotting_data %>% 
                filter(model_full==vals[1], noise==vals[2]) %>%
                select(real_alpha, contains(paste0("metric_", vals[3]))) %>%
                gather(order, metric, 2:3)
            p <- ggplot(pd, aes(x=real_alpha, y=metric)) + 
                geom_line(aes(color=order)) +
                labs(x="Real Number of Samples", y=vals[3]) +
                ggtitle(as.character(vals))
            ggsave(paste0(out_path, "/exp_min__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".png"), p)
            
            vals = c(vals[[1]], vals[[2]], metric)
            pd <- plotting_data %>% 
                filter(model_full==vals[1], noise==vals[2]) %>%
                select(real_alpha, contains(paste0("synth_alpha_", vals[3]))) %>%
                gather(order, metric, 2:3)
            p <- ggplot(pd, aes(x=real_alpha, y=metric)) + 
                geom_line(aes(color=order)) +
                labs(x="Real Number of Samples", y=vals[3]) +
                ggtitle(as.character(vals))
            ggsave(paste0(out_path, "/exp_min_alpha__model_", vals[1], "__noise_", vals[2], "__metric_", vals[3], ".png"), p)

        }

    }

}

spaghetti_plots(data_without_reals)

spaghetti_plots <- function(df) {
    spag_data <- df %>% 
        inner_join(min_data) %>% 
        inner_join(exp_data) %>% 
        inner_join(min_exp_data) %>% 
        inner_join(exp_min_data)

    for (i in 1:nrow(distinct(select(spag_data, c(model_full, real_alpha, noise))))) {

        vals <- distinct(select(data, c(model_full, real_alpha, noise)))[i,]
        fd <- spag_data %>% filter(model_full == vals[[1]], real_alpha == vals[[2]], noise == vals[[3]])
        for (metric in metrics) {
        
            p <- ggplot(fd) +
                geom_line(aes_string(x="synth_alpha", y=metric, group="seed"), alpha=0.05) +
                geom_line(aes_string(x="synth_alpha", y=paste0("exp_", metric))) +
                geom_hline(aes_string(yintercept=paste0("exp_min_metric_", metric), color='"Expected Minimum"')) +
                geom_vline(aes_string(xintercept=paste0("exp_min_synth_alpha_", metric), color='"Expected Minimum"')) +
                geom_hline(aes_string(yintercept=paste0("min_exp_metric_", metric), color='"Minimum Expected"')) +
                geom_vline(aes_string(xintercept=paste0("min_exp_synth_alpha_", metric), color='"Minimum Expected"'))
                # geom_smooth(aes_string(x="synth_alpha", y=metric))
            ggsave(paste0(out_path, "/spag__model_", vals[[1]], "__real_alpha_", vals[[2]], "__noise_", vals[[3]], "__metric_", metric, ".png"), p)
        
        }

    }
}




epsilon_demo(df)

epsilon_demo <- function(df) {

    grouped_data <- df %>% 
        group_by(epsilon, real_alpha, synth_alpha) %>%
        summarise(ll = mean(ll), auc = mean(auc), param_mse = mean(param_mse))

    real_baseline_100 <- grouped_data %>% filter(real_alpha == 1.0, synth_alpha == 0)
    real_baseline_10 <- grouped_data %>% filter(real_alpha == 0.1, synth_alpha == 0)
    synth_total_100 <- grouped_data %>% filter(synth_alpha == 1.0, real_alpha == 0)
    synth_total_10 <- grouped_data %>% filter(synth_alpha == 0.1, real_alpha == 0)
    synth_best <- grouped_data %>%
        filter(real_alpha == 0, synth_alpha > 0) %>%
        group_by(epsilon) %>%
        summarise(ll = min(ll), auc = min(auc), param_mse = min(param_mse))

    breaks <- 10^(-2:3)
    minor_breaks <- rep(1:9, 6)*(10^rep(-2:3, each=9))

    for (metric in metrics) {
        p <- ggplot() +
            geom_line(data = real_baseline_100, aes(x=epsilon, y=!!sym(metric), color="N = 100", linetype="Real")) +
            geom_line(data = real_baseline_10, aes(x=epsilon, y=!!sym(metric), color="N = 10", linetype="Real")) +
            geom_line(data = synth_total_100, aes(x=epsilon, y=!!sym(metric), color="N = 100", linetype="Synthetic")) +
            geom_line(data = synth_total_10, aes(x=epsilon, y=!!sym(metric), color="N = 10", linetype="Synthetic")) +
            geom_line(data = synth_best, aes(x=epsilon, y=!!sym(metric), color="N = Optimal", linetype="Synthetic")) +
            scale_x_log10(breaks = breaks, minor_breaks = minor_breaks) +
            # scale_y_log10(breaks = breaks, minor_breaks = minor_breaks) +
            theme_light() +
            annotation_logticks()
        ggsave(paste0(out_path, "/noise_demo_gaussian___metric_", metric, ".png"), p, height=4, width=7, dpi=320)
    }
}