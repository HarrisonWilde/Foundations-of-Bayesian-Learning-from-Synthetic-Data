
function plot_roc_curve(ys, ps)
    tprs, fprs, _ = roc_curve([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
    display(plot(tprs, fprs))
end


function plot_all(df, real_αs, synth_αs, divergences, metrics, t)

    plot_all_αs(df, real_αs, synth_αs, divergences, metrics, t)
    plot_all_divergences(df, divergences, metrics, t)

end


function plot_all_αs(df, real_αs, synth_αs, divergences, metrics, t)

    for metric in metrics
        for α in real_αs
            plot_real_α(df, α, divergences, metric, t)
        end
        for α in synth_αs
            plot_synth_α(df, α, divergences, metric, t)
        end
    end

end


function plot_real_α(df, α, divergences, metric, t)

    mkpath("src/logistic_regression/plots/$(t)/")
    fdf = filter(row -> row[:real_α] == α, df)
    p = @df fdf plot(
        :synth_α,
        [cols(Symbol("$(div)_$(metric)_mean") for div in divergences)],
        ribbon = [cols(Symbol("$(div)_$(metric)_std") for div in divergences)],
        title = "$(metric) divergence comparison, real alpha = $(α)",
        label = [div for div in divergences],
        xlabel = "Synthetic Alpha",
        ylabel = metric,
        legendtitle = "Divergence",
    )
    p = plot!(size=(1000, 700), legend=:outertopright)
    png(p, "src/logistic_regression/plots/$(t)/real_alpha_$(α)__$(metric)")

end


function plot_synth_α(df, α, divergences, metric, t)

    mkpath("src/logistic_regression/plots/$(t)/")
    p = @df filter(row -> row[:synth_α] == α, df) plot(
        :real_α,
        [cols(Symbol("$(div)_$(metric)_mean") for div in divergences)],
        ribbon = [cols(Symbol("$(div)_$(metric)_std") for div in divergences)],
        title = "$(metric) divergence comparison, synth alpha = $(α)",
        label = [div for div in divergences],
        xlabel = "Real Alpha",
        ylabel = metric,
        legendtitle = "Divergence",
    )
    p = plot!(size=(1000, 700), legend=:outertopright)
    png(p, "src/logistic_regression/plots/$(t)/synth_alpha_$(α)__$(metric)")

end


function plot_all_divergences(df, divergences, metrics, t)

    for metric in metrics
        for divergence in divergences
            plot_divergences(df, divergence, metric, t)
        end
    end

end


function plot_divergences(df, divergence, metric, t)

    mkpath("src/logistic_regression/plots/$(t)/")
    p = @df df plot(
        :real_α + :synth_α,
        [cols(Symbol("$(divergence)_$(metric)_mean"))],
        ribbon = [cols(Symbol("$(divergence)_$(metric)_std"))],
        title = divergence,
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = metric,
    )
    p = @df filter(row -> row[:synth_α] == 0.0, df) plot!(
        :real_α,
        [cols(Symbol("$(divergence)_$(metric)_mean"))],
        ribbon = [cols(Symbol("$(divergence)_$(metric)_std"))],
        label = "varying"
    )
    p = plot!(size=(1000, 700), legend=:outertopright)
    png(p, "src/logistic_regression/plots/$(t)/$(divergence)__$(metric)")

end
