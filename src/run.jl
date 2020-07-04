using Pkg; Pkg.activate(".")
using ArgParse
using Distributed
# using ClusterManagers
# # # addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])))
# println("We are all connected and ready.")
# for i in workers()
#     host, pid = fetch(@spawnat i (gethostname(), getpid()))
#     println(host, pid)
# end
include("common/utils.jl")
include("common/init.jl")

function main()

    args = parse_cl()
    λs, K, algorithm = args["scales"], args["num_repeats"], args["algorithm"]
    path, dataset, label, ε = args["path"], args["dataset"], args["label"], args["epsilon"]
    iterations, folds, split = args["iterations"], args["folds"], args["split"]
    use_ad, distributed, sampler, no_shuffle = args["use_ad"], args["distributed"], args["sampler"], args["no_shuffle"]
    experiment_type = args["experiment"]

    if experiment_type == "gaussian"
        include("gaussian/run_golden.jl")
    elseif experiment_type == "regression"
        include("regression/run.jl")
    elseif experiment_type == "logistic_regression"
        include("logistic_regression/run.jl")
    end
    run_experiment()

end

main()
