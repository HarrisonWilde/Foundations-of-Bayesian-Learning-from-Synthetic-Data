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
dropmissing!(results)
gdf = groupby(results, [:scale, :model_name, :real_n, :synth_n])
for g in gdf

    display(density(g[:ll]))
    display(density(g[:kld]))
    display(density(g[:wass]))

end
