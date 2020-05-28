# Synthetic Data Experiments with Bayesian Inference

*All of the code is written by Harrison Wilde with help from Jack Jewson on the Stan models and general advice from Sebastian Vollmer.*

## Dependencies

See the Julia `Manifest.toml` and `Project.toml` for a full list of dependencies to run files found in `src`, you can run `Pkg.activate(".")` or start Julia with the `--project=@.` option at the top of this repo to first activate the project file. Then run `Pkg.instantiate()` to install all of the dependencies. The Python code is somewhat deprecated in favour of Julia which is much more powerful and performant in the tasks laid out; I will provide a full list of dependencies for the Python code at some point, or on request, though they are fairly standard.

To use the `CmdStan` sampler option, a [working installation of CmdStan is required](https://mc-stan.org/users/interfaces/cmdstan), preferably version 2.23.0 but any later version should also work in lieu of breaking changes.

To generate your own synthetic data using the same mechanism as us, we provide `run_gan.py` for the running of the PATE-GAN (paper [here](https://openreview.net/pdf?id=S1zk9iRqF7), source code [here](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/4fb84b06c83b7ed80b681c9b7d91e66c78495378/alg/pategan/)). Very little was changed in the workings of the GAN itself, we simply provide an interface for running it and a means to discretise some of its outputs. TensorFlow, Pandas and Numpy are required alongside a recent Python version to run this code (tested on 3.7.6).

## Usage and Reproducing the Results in Our Paper

Run `julia src/<experiment_type>/run.jl -h` to see options, then specify options such as:

```
julia src/logistic_regression/run.jl \
    --path <path_to_top_of_repo> \
    --dataset uci_heart \
    --label target \
    --epsilon 6.0 \
    --iterations 100 \
    --folds 5 \
    --sampler AHMC \
    --distributed
```

To reproduce the UCI Heart logistic regression results shown in the paper. Prior to this the synthetic data must be generated using the PATE-GAN mentioned in the paper, this can be done by executing:

```
python run_gan.py -i uci_heart --targets target --no_split --iter 10000 --teachers 75
```

It is likely that I will make the exact synthetic datasets available in some way to go along with this codebase, removing the need for a user to generate these datasets themselves.

Note that the `iterations` option is to facilitate the running of this code numerous times to account for noise in the MCMC process and stochasticity introduced through carrying out cross validation in the Bayesian setting, it is not required for general use of this code. Similarly, `distributed` allows for the code to run on a cluster using the SLURM workload manager, allowing for massive parallelisation of tasks amongst the code.

## Known Issues

There is a requirement that a user must edit `matplotlib`'s `font_manager.py` to get multiprocessing to work correctly in the Python code on Mac OS Catalina. This is an issue on Python's end, not with the code, so no fix is possible on my end until they act upon the issue raised. Change:
```
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_get_font.cache_clear)
```

To:
```
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_get_font.cache_clear)
```

## Acknowledgements

I'd like to extend special thanks to the great groups of people managing  `Turing.jl`, `Stan.jl`, `MLJ.jl` and `AdvancedHMC.jl` for being so helpful in my queries throughout. Especially to Thibaut Lienart, Hong Ge, David Widmann, Mohamed Tarek, Cameron Pfiffer, Robert Goedman, Tor Fjelde, Martin Trapp who all spared their time in helping with bugs I uncovered, performance issues or other pain points throughout this research.

The PATE-GAN original source code can be found [here]() alongside their [paper]()
