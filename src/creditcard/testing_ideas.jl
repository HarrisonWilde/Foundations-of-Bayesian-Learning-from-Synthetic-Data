using Distributed
addprocs(4)

@everywhere begin
    using Pkg; Pkg.activate(".")
    using ProgressMeter
end

p = Progress(10)
progress_pmap(1:10, progress=p) do x
    sleep(1.0)
    x^2
end

using ProgressMeter
using Distributed
using SharedArrays

addprocs(6)
p = Progress(100)
channel = RemoteChannel(()->Channel{Bool}(100), 1)
results = SharedArray{Float64, 1}(100)

@sync begin
    # this task prints the progress bar
    @async while take!(channel)
        next!(p)
    end

    # this task does the computation
    @async begin
        @distributed for i in 1:100
            sleep(0.1)
            put!(channel, true)
            results[i] = i^2
        end
        put!(channel, false) # this tells the printing task to finish
    end
end
