function init_stan_models(sampler, model_names, experiment_type, target_acceptance_rate, nchains, n_samples, n_warmup; dist = true)

    tmpdir = dist ? "$(@__DIR__)/tmp/" : mktempdir()
    if sampler == "Stan"
        models = [(
            "$(name)_$(myid())",
            SampleModel(
                "$(name)_$(myid())",
                open(
                    f -> read(f, String),
                    "src/$(experiment_type)/stan/$(name)_$(experiment_type).stan"
                ),
                n_chains = nchains,
                tmpdir = tmpdir,
                method = StanSample.Sample(
                    num_samples=n_samples - n_warmup,
                    num_warmup=n_warmup,
                    adapt=StanSample.Adapt(delta=target_acceptance_rate)
                )
            )
        ) for name in model_names]
    elseif sampler == "CmdStan"
        models = [(
            "$(name)_$(myid())",
            Stanmodel(
                CmdStan.Sample(
                    num_samples=n_samples - n_warmup,
                    num_warmup=n_warmup,
                    adapt=CmdStan.Adapt(delta=target_acceptance_rate)
                );
                name = "$(name)_$(myid())",
                nchains = nchains,
                model = open(
                    f -> read(f, String),
                    "src/$(experiment_type)/stan/$(name)_$(experiment_type).stan"
                ),
                tmpdir = tmpdir,
                output_format = :mcmcchains
            )
        ) for name in model_names]
    end
    return models

end
