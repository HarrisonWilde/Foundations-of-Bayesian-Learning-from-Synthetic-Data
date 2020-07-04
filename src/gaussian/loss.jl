function βloss(μ, σ, β, y)
	loss = @. (1 / β) * pdf_N(μ, abs(σ), y) ^ β - int_term(abs2(σ), β)
	return -sum(loss)
end
