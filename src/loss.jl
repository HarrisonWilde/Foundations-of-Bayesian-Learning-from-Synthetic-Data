function gaussian_βloss(μ, σ, β, y)
	loss = @. (1 / β) * pdf_N(μ, abs(σ), y) ^ β - int_term(abs2(σ), β)
	return -sum(loss)
end


function logistic_βloss(X, y, β, θ)

    Xθ = X * θ
    loss = @. 1 / β * logistic(y .* Xθ) ^ β - 1 / (β + 1) * (logistic(Xθ) ^ (β + 1) + logistic(Xθ) ^ (β + 1))
    return -sum(loss)

end


function logistic_yXβ(yX, β, θ)

    eyXθ = exp.(-yX * θ)
    return yX' * (eyXθ ./ (1 .+ eyXθ) .^ β)

end
function logistic_∇βloss(yX, β, θ)

    return -yXβ(yX, β + 1, θ) + yXβ(yX, β + 2, θ) + yXβ(-yX, β + 2, θ)

end


function logistic_βyXᵀyX(yX, β, θ)

    eyXθ = exp.(-yX * θ)
    return (yX)' * (yX) .* (
        (β .* eyXθ .^ 2) ./ ((1 .+ eyXθ) .^ (β + 1))
        - (eyXθ ./ (1 .+ eyXθ) .^ β)
    )

end
function logistic_Hβloss(yX, β, θ)

    return -βyXᵀyX(yX, β + 1, θ) + βyXᵀyX(yX, β + 2, θ) + βyXᵀyX(-yX, β + 2, θ)

end
