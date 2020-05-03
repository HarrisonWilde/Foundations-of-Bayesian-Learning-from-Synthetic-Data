
function plot_roc_curve(ys, ps)
    tprs, fprs, _ = roc_curve([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
    display(plot(tprs, fprs))
end


function plot_all(df, real_αs, t)

    plot_all_αs(df, real_αs, t)
    plot_divergences(df, t)

end


function plot_all_αs(df, real_αs, t)

    for α in real_αs
        plot_α(df, α, t)
    end

end


function plot_α(df, α, t)

    p = @df filter(row -> row[:real_α] == α, df) plot(
        :synth_α,
        [:β :weighted :naive :no_synth],
        title = "Real Alpha = $(α)",
        xlabel = "Synthetic Alpha",
        ylabel = "ROC AUC",
        legendtitle = "Divergence",
    )
    png(p, "src/creditcard/plots/$(t)_alpha$(α)")

end


function plot_divergences(df, t)

    p_β = @df df plot(
        :real_α + :synth_α,
        [:β],
        title = "Beta-Divergence",
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = "ROC AUC",
    )
    p_β = @df filter(row -> row[:synth_α] == 0.0, df) plot!(:real_α, [:β], label = "varying")
    p_weighted = @df df plot(
        :real_α + :synth_α,
        [:weighted],
        title = "KLD Weighted",
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = "ROC AUC"
    )
    p_weighted = @df filter(row -> row[:synth_α] == 0.0, df) plot!(:real_α, [:weighted], label = "varying")
    p_naive = @df df plot(
        :real_α + :synth_α,
        [:naive],
        title = "KLD Naive",
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = "ROC AUC"
    )
    p_naive = @df filter(row -> row[:synth_α] == 0.0, df) plot!(:real_α, [:naive], label = "varying")
    p_no_synth = @df df plot(
        :real_α + :synth_α,
        [:no_synth],
        title = "KLD No Synthetic",
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = "ROC AUC"
    )
    p_no_synth = @df filter(row -> row[:synth_α] == 0.0, df) plot!(:real_α, [:no_synth], label = "varying")

    png(p_β, "src/creditcard/plots/$(t)_beta")
    png(p_weighted, "src/creditcard/plots/$(t)_weighted")
    png(p_naive, "src/creditcard/plots/$(t)_naive")
    png(p_no_synth, "src/creditcard/plots/$(t)_no_synth")

end
