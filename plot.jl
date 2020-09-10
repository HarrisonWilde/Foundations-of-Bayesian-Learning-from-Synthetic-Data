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

function mean_std(x)
    std(x) / sqrt(length(x))
end

function concat(result)
    if result.model == "beta"
        string(result.model, "_", result.weight, "_", result.beta)
    elseif result.model == "weighted"
        string(result.model, "_", result.weight)
    else 
        result.model
    end
end

# args = Dict(
#     "path" => "from_cluster/gaussian/outputs/Pull1/out.csv",
#     "x_axis" => "synth_n",
#     "group_by" => "model",
#     "loop" => ["noise", "real_n"],
#     "iter" => [],
#     "noise" => [],
#     "weight" => [],
#     "beta" => [],
#     "model" => [],
#     "real_n" => [],
#     "synth_n" => [],
#     "metrics" => ["ll", "kld", "wass"]
# )

function main()

    args = parse_cl()
    print(split(args["path"], "/")[end][1:end-4])
    results = CSV.read(args["path"], copycols=true)
    @show args
    dropmissing!(results)
    # mapcols(col -> replace!(col, -1.0=>NaN), results)

    # Filter data according to flags
    for f ∈ ["iter", "noise", "model", "weight", "beta", "real_n", "synth_n"]
        if length(args[f]) > 0
            results = filter(row -> row[Symbol(f)] ∈ args[f], results)
        end
    end
    
    if args["x_axis"] == "n"
        sort!(results, [:real_n, :synth_n])
    else
        sort!(results, [Symbol(args["x_axis"])])
    end
    if args["x_axis"] == "n"
        gdf = groupby(results, unique(vcat([:real_n, :synth_n], [Symbol(l) for l in args["loop"]])))
        add_ns = true
    elseif args["group_by"] == "model"
        results.model_full = map(row -> concat(row), eachrow(results))
        gdf = groupby(results, unique(vcat([:model_full, Symbol(args["x_axis"])], [Symbol(l) for l in args["loop"]])))
        add_ns = false
    else
        gdf = groupby(results, unique(vcat([Symbol(args["group_by"]), Symbol(args["x_axis"])], [Symbol(l) for l in args["loop"]])))
        add_ns = false
    end
    
    df = combine(gdf, vcat(
        [Symbol(metric) => mean for metric in args["metrics"]],
        [Symbol(metric) => mean_std for metric in args["metrics"]])
    )

    uniques = unique(df[args["loop"]])

    L = [
        (metric, Tuple(conf), join(["$(args["loop"][i])$(Tuple(conf)[i])" for i in 1:size(conf)[1]], "_"))
        for metric ∈ args["metrics"]
        for conf ∈ eachrow(uniques)
    ]

    plot_path = join([replace(
        join(split(args["path"], "/")[1:end-1], "/"),
        "outputs"=>"plots"
    ), split(args["path"], "/")[end][1:end-4]], "/")
    
    mkpath(plot_path)

    for (metric, conf, title) ∈ L

        @show metric, conf, title

        fdf = filter(row -> Tuple(row[args["loop"]]) == conf, df)
        if add_ns

            p = @df fdf plot(
                :real_n + :synth_n,
                [cols(Symbol("$(metric)_mean"))],
                ribbon = [cols(Symbol("$(metric)_mean_std"))],
                fillalpha = 0.1,
                title = """$(args["x_axis"])_against_$(metric)_$(title)""",
                group = :real_n,
                legendtitle = "Real Samples",
                xlabel = "Total Samples",
                ylabel = metric,
                # yscale = (metric == "ll") ? :log10 : :identity
            )
            p = @df filter(row -> row[:synth_n] == 0.0, fdf) plot!(
                :real_n,
                [cols(Symbol("$(metric)_mean"))],
                ribbon = [cols(Symbol("$(metric)_mean_std"))],
                fillalpha = 0.2,
                label = "varying",
                # yscale = (metric == "ll") ? :log10 : :identity
            )
            p = plot!(size=(1000, 700), legend=:outertopright)

        elseif !isnothing(args["group_by"])

            p = @df fdf plot(
                cols(Symbol(args["x_axis"])),
                [cols(Symbol("$(metric)_mean"))],
                ribbon = [cols(Symbol("$(metric)_mean_std"))],
                fillalpha = 0.2,
                title = """$(args["x_axis"])_against_$(metric)_$(title)""",
                group = args["group_by"] == "model" ? :model_full : args["group_by"],
                # label = [div for div in divergences],
                xlabel = args["x_axis"],
                ylabel = metric,
                legendtitle = args["group_by"],
                # yscale = (metric == "ll") ? :log10 : :identity
            )
            p = plot!(size=(1000, 700), legend=:outertopright)

        else

            p = @df fdf plot(
                cols(Symbol(args["x_axis"])),
                [cols(Symbol("$(metric)_mean"))],
                ribbon = [cols(Symbol("$(metric)_mean_std"))],
                fillalpha = 0.2,
                title = """$(args["x_axis"])_against_$(metric)_$(title)""",
                # label = [div for div in divergences],
                xlabel = args["x_axis"],
                ylabel = metric,
                # yscale = (metric == "ll") ? :log10 : :identity
            )
            p = plot!(size=(1000, 700), legend=:outertopright)

        end
        png(p, """$(plot_path)/$(args["x_axis"])_against_$(metric)_$(title)""")

    end

    println("Done plotting")

end

main()