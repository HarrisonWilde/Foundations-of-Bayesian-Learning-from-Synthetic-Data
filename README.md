# Foundations of Bayesian Learning from Synthetic Data

*All of the code is written by Harrison Wilde with help from Jack Jewson on the Stan models and general advice from Sebastian Vollmer.*

## Paper Abstract

There is significant growth and interest in the use of synthetic data as an enabler for machine learning in environments where the release of real data is restricted due to privacy or availability constraints. Despite a large number of methods for synthetic data generation, there are comparatively few results on the statistical properties of models learnt on synthetic data, and fewer still for situations where a researcher wishes to augment real data with another party's synthesised data. We use a Bayesian paradigm to characterise the updating of model parameters when learning in these settings, demonstrating that caution should be taken when applying conventional learning algorithms without appropriate consideration of the synthetic data generating process and learning task. Recent results from general Bayesian updating support a novel and robust approach to Bayesian synthetic-learning founded on decision theory that outperforms standard approaches across repeated experiments on supervised learning and inference problems.

## Dependencies

See the Julia `Manifest.toml` and `Project.toml` for a full list of dependencies to run files found in `src`, you can run `Pkg.activate(".")` or start Julia with the `--project=@.` option at the top of this repo to first activate the project file. Then run `Pkg.instantiate()` to install all of the dependencies. The Python code is somewhat deprecated in favour of Julia which is much more powerful and performant in the tasks laid out; I will provide a full list of dependencies for the Python code at some point, or on request, though they are fairly standard.

To use the `CmdStan` sampler option, a [working installation of CmdStan is required](https://mc-stan.org/users/interfaces/cmdstan), preferably version 2.23.0 but any later version should also work in lieu of breaking changes.

To generate your own synthetic data using the same mechanism as us, we provide `run_gan.py` for the running of the PATE-GAN \[1\] (paper [here](https://openreview.net/pdf?id=S1zk9iRqF7), source code [here](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/4fb84b06c83b7ed80b681c9b7d91e66c78495378/alg/pategan/)). Very little was changed in the workings of the GAN itself, we simply provide an interface for running it and a means to discretise some of its outputs as well as some specific alterations to clean up the datasets we used. TensorFlow, Pandas and Numpy are required alongside a recent Python version to run this code (tested on 3.7.6).

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

To reproduce the UCI Heart logistic regression results shown in the paper. Prior to this, the synthetic data must be generated using the PATE-GAN mentioned in the paper, this can be done by executing:

```
python run_gan.py -i uci_heart --targets target --no_split --iter 10000 --teachers 75
```

It is likely that I will make the exact synthetic datasets available in some way to go along with this codebase, removing the need for a user to generate these datasets themselves.

Note that the `iterations` option is to facilitate the running of this code numerous times to account for noise in the MCMC process and stochasticity introduced through carrying out cross validation in the Bayesian setting, it is not required for general use of this code. Similarly, `distributed` allows for the code to run on a cluster using the SLURM workload manager, allowing for massive parallelisation of tasks amongst the code.

## Plotting Results

There is a relatively flexible framework for plotting the results of our experiments. That framework is includede in this repository in the interest of transparency and to hopefully encourage further exploration with different models and datasets. Some example plotting commands:

- To plot what we call a "branching" plot illustrating model performance as synthetic data varies under various fixed real data amounts.

```
julia --project=@. plot.jl \
--path "from_cluster/gaussian/outputs/final_csvs/grid1.csv" \
--x_axis n \
--loop noise weight beta model
```

- To plot log loss and KLD metrics against the amount of synthetic data across all model configurations

```
julia --project=@. plot.jl \
--path "from_cluster/gaussian/outputs/final_csvs/grid1.csv" \
--x_axis synth_n \
--loop noise real_n \
--metrics ll kld
--group_by model
```

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

I'd like to extend special thanks to the great groups of people managing the [Turing Language](https://github.com/TuringLang) \[2\], [StanJulia](https://github.com/StanJulia) \[3\] and [MLJ](https://github.com/alan-turing-institute/MLJ.jl) \[4\] for being so helpful. Especially to Thibaut Lienart, Hong Ge, David Widmann, Mohamed Tarek, Cameron Pfiffer, Robert Goedman, Tor Fjelde, Martin Trapp who all spared their time in helping with questions, bugs, performance issues or other pain points throughout. Additional thanks to the Stan and Julia communities in general for enabling such excellent environments for research.

## Citing this work

<Reference to follow>
    
## References

\[1\] Jordon, James, Jinsung Yoon, and Mihaela van der Schaar. "PATE-GAN: Generating synthetic data with differential privacy guarantees." International Conference on Learning Representations. 2018.

\[2\] Ge, Hong, Kai Xu, and Zoubin Ghahramani. "Turing: A language for flexible probabilistic inference." (2018).

\[3\] Carpenter, Bob, et al. "Stan: A probabilistic programming language." Journal of statistical software 76.1 (2017).

\[4\] Blaom, Anthony D., et al. "MLJ: A Julia package for composable Machine Learning." arXiv preprint arXiv:2007.12285 (2020).
