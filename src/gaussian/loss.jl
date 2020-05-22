function βloss(μ, σ², β, y)
	loss = @. (1 / β) * pdf_N(μ, √σ², y) ^ β - int_term(σ², β)
	return -sum(loss)
end
