using CSV
using DataFrames
using Plots
using StatsPlots
using Statistics
using ArgParse
# include("src/plotting.jl")
theme(:vibrant)

function parse_cl()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--path"
            help = "specify the path to the csv to plot from"
            arg_type = String
            default = "."
        "--x_axis", "-x"
            required = true
            arg_type = String
        "--group_by", "-g"
            arg_type = String
        "--loop", "-l"
            nargs = '+'
            arg_type = String
        # Filters
        "--iter"
            nargs = '*'
            arg_type = Int
        "--noise"
            nargs = '*'
            arg_type = Float64
        "--model"
            nargs = '*'
            arg_type = String
        "--weight"
            nargs = '*'
            arg_type = Float64
        "--beta"
            nargs = '*'
            arg_type = Float64
        "--real_n"
            nargs = '*'
            arg_type = Int
        "--synth_n"
            nargs = '*'
            arg_type = Int
        "--metrics"
            nargs = '+'
            arg_type = String
            default = ["ll", "kld", "wass"]
    end
    return parse_args(s)
end

# Plot x against y, we are interested in seeing how the expected minimum compares to the minimum of the expected curve

function main()

    

end

main()