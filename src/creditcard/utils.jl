"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
end

"""
Derivative of the logistic function
"""
function ∂logistic(z::Float64)
	a = exp(z) + 1
    return (1 / a) - (1 / (a ^ 2))
end


function ∂pdf∂θ(X::Array{Float64,2}, y::Array{Int64,1}, z::Array{Float64,1}, θ::Array{Float64,1})
    return transpose(X) * (y .* ∂logistic.(z)) - transpose(X) * ((1.0 .- y) .* ∂logistic.(z))
end