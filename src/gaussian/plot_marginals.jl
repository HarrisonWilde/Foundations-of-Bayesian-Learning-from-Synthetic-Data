using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
include("../common/plotting.jl")
theme(:vibrant)

function mean_std(x)
    std(x) / sqrt(length(x))
end

t = "21_32_59__27_05_2020_AHMC"
results = CSV.read("src/gaussian/outputs/$(t)/1_out.csv", copycols=true)
mkpath("src/gaussian/plots/$(t)")
dropmissing!(results)
gdf = groupby(results, [:scale, :model_name, :real_n, :synth_n])
for g in gdf

    title = "scale: $(g[1, :scale]), model: $(g[1, :model_name]), real_n: $(g[1, :real_n]), synth_n: $(g[1, :synth_n])"
    d = histogram(g[:ll], normalize=:pdf)
    density!(g[:ll], title = "ll $(title)")
    display(d)
    png(d, "src/gaussian/plots/$(t)/ll_$(title)")
    d = histogram(g[:kld], normalize=:pdf)
    density!(g[:kld], title = "kld $(title)")
    display(d)
    png(d, "src/gaussian/plots/$(t)/kld_$(title)")
    d = histogram(g[:wass], normalize=:pdf)
    density!(g[:wass], title = "wass $(title)")
    display(d)
    png(d, "src/gaussian/plots/$(t)/wass_$(title)")

end
