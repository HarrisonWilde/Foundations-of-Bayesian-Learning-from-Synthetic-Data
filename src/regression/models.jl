@model logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, w) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ WeightedBernoulliLogit.(X_real * θ, 1)
	y_synth .~ WeightedBernoulliLogit.(X_synth * θ, w)

end
