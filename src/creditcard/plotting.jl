
function plot_roc_curve(ys, ps)
    tprs, fprs, _ = roc_curve([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
    display(plot(tprs, fprs))
end


function plot_all(df, real_αs, divergences, metrics, t)

    plot_all_αs(df, real_αs, divergences, metrics, t)
    plot_all_divergences(df, divergences, metrics, t)

end


function plot_all_αs(df, real_αs, divergences, metrics, t)

    for metric in metrics
        for α in real_αs
            plot_α(df, α, divergences, metric, t)
        end
    end

end


function plot_α(df, α, divergences, metric, t)

    mkpath("src/creditcard/plots/$(t)/")
    p = @df filter(row -> row[:real_α] == α, df) plot(
        :synth_α,
        [cols(Symbol("$(div)_$(metric)") for div in divergences)],
        title = "$(metric) divergence comparison, real alpha = $(α)",
        label = [div for div in divergences],
        xlabel = "Synthetic Alpha",
        ylabel = metric,
        legendtitle = "Divergence",
    )
    p = plot!(size=(1000, 700), legend=:outertopright)
    png(p, "src/creditcard/plots/$(t)/real_alpha_$(α)__$(metric)")

end


function plot_all_divergences(df, divergences, metrics, t)

    for metric in metrics
        for divergence in divergences
            plot_divergences(df, divergence, metric, t)
        end
    end

end


function plot_divergences(df, divergence, metric, t)

    mkpath("src/creditcard/plots/$(t)/")
    p = @df df plot(
        :real_α + :synth_α,
        [cols(Symbol("$(divergence)_$(metric)"))],
        title = divergence,
        group = :real_α,
        legendtitle = "Real Alpha",
        xlabel = "Real + Synthetic Alpha",
        ylabel = metric,
    )
    p = @df filter(row -> row[:synth_α] == 0.0, df) plot!(:real_α, [cols(Symbol("$(divergence)_$(metric)"))], label = "varying")
    p = plot!(size=(1000, 700), legend=:outertopright)
    png(p, "src/creditcard/plots/$(t)/$(divergence)__$(metric)")

end
