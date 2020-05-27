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

t = "kag_cervical_cancer_15_07_19__24_05_2020"
results = CSV.read("src/logistic_regression/outputs/$(t)/out.csv", copycols=true)
dropmissing!(results)
# bayes_factors = load("src/creditcard/outputs/bayes_factors___$(t).jld")["data"]
sort!(results, [:real_α, :synth_α])
real_αs = unique(results[!, :real_α])
synth_αs = unique(results[!, :synth_α])
divergences, metrics = ["beta" "weighted" "naive" "no_synth"], ["ll" "auc"]
gdf = groupby(results, [:real_α, :synth_α])
# size(gdf[i] for i in 1:length(gdf))
df = combine(gdf, vcat(
    [Symbol("$(div)_$(metric)") => mean for div in divergences for metric in metrics],
    [Symbol("$(div)_$(metric)") => mean_std for div in divergences for metric in metrics]))
mapcols(col -> replace!(col, NaN=>0), df)
# t = names(df, r"_ll_mean")
# tdf = transform(df, t .=> ByRow(log))
plot_all(df, real_αs, synth_αs, divergences, metrics, t)
println("Done plotting")
