using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
include("../common/plotting.jl")
theme(:vibrant)


t = "uci_heart_16_02_03__13_05_2020"
results = CSV.read("src/logistic_regression/outputs/$(t)_out.csv", copycols=true)
# bayes_factors = load("src/creditcard/outputs/bayes_factors___$(t).jld")["data"]
sort!(results, (:real_α, :synth_α))
real_αs = unique(results[!, :real_α])
synth_αs = unique(results[!, :synth_α])
divergences, metrics = ["beta" "weighted" "naive"], ["auc" "ll"]
gdf = groupby(results, [:real_α, :synth_α])
df = combine(gdf, vcat([Symbol("$(div)_$(metric)") => mean for div in divergences for metric in metrics], [Symbol("$(div)_$(metric)") => std for div in divergences for metric in metrics]))
mapcols(col -> replace!(col, NaN=>0), df)
plot_all(df, real_αs, synth_αs, divergences, metrics, t)
println("Done plotting")

# plot_real_α(df, 0.025, divergences, metrics[1], t)
