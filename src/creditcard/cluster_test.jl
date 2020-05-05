@everywhere begin
    using Pkg; Pkg.activate(".")
    using LinearAlgebra
    using ForwardDiff
end

function f(M::Int, N::Int)
    e = @distributed (+) for n = 1:N
        A = rand(M, M)
        A = ForwardDiff.gradient(x ->  0.5x[1] + rand(), [4])[1] * (A + A')
        sort(real(eigvals(A)))
    end
    return e / N
end

# compile
f(10,10)
sleep(1.0)

# run with user parameters
@assert length(ARGS) == 2
M = parse(Int, ARGS[1])
N = parse(Int, ARGS[2])

@time out = f(M, N)
println(out)
sleep(1.0)
