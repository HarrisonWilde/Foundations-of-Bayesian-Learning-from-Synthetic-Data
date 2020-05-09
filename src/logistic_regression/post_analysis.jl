using CSV
using DataFrames
using Plots
using StatsPlots
include("src/creditcard/plotting.jl")
theme(:vibrant)


t = "17_48_15__04_05_2020"
results = CSV.read("src/creditcard/outputs/results___$(t).csv", copycols=true)
# bayes_factors = load("src/creditcard/outputs/bayes_factors___$(t).jld")["data"]
sort!(results, (:real_α, :synth_α))
real_αs = unique(results[!, :real_α])
synth_αs = unique(results[!, :synth_α])
plot_all(results, real_αs, synth_αs, ["beta" "weighted" "naive" "no_synth"], ["auc", "ll"], t)
