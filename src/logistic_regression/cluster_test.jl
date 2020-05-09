@everywhere begin
    using Pkg; Pkg.activate(".")
    using LinearAlgebra
    using ForwardDiff
    using ProgressMeter
end

function f(M::Int, N::Int)
    outs = progress_pmap(1:N, progress=Progress(N)) do n
        A = rand(M, M)
        A = ForwardDiff.gradient(x ->  0.5x[1] + rand(), [4])[1] * (A + A')
        return sort(real(eigvals(A))), ["Hi$(n+1)", "Hi$(n)"]
    end
    return outs
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
