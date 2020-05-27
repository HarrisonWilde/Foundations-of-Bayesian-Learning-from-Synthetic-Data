using Statistics
using HypothesisTests

invϕ = (√5 - 1) / 2
invϕ² = (3 - √5) / 2

function g(n::Int64)
    -2 * log(n) + log(n / 15) ^2 + 0.1 * randn()
end

function gss(g, xmin, xmax, gtol=0.2, tol=1e-5, C=10)

    (xmin, xmax) = sort([xmin, xmax])
    h = xmax - xmin
    if h <= tol
        return mean([xmin, xmax]), NaN
    end

    p = plot(xmin:xmax, g.(xmin:xmax))

    n = ceil(Int, log(tol / h) / log(invϕ))
    xmid₁ = round(Int, xmin + invϕ² * h)
    xmid₂ = round(Int, xmin + invϕ * h)

    gall = []

    gdiffs = zeros(0)
    gmins = zeros(0)
    gmid₁s = zeros(0)
    gmid₂s = zeros(0)
    gmaxs = zeros(0)
    for i ∈ 1:C
        gmin = g(xmin)
        gmid₂ = g(xmid₂)
        gmid₁ = g(xmid₁)
        gmax = g(xmax)
        append!(gmins, gmin)
        append!(gmid₁s, gmid₁)
        append!(gmid₂s, gmid₂)
        append!(gmaxs, gmax)
        append!(gall, [(xmin, gmin), (xmid₁, gmid₁), (xmid₂, gmid₂), (xmax, gmax)])

        gdiff = gmid₂ - gmid₁
        append!(gdiffs, gdiff)

    end
    ci = confint(OneSampleTTest(gdiffs))

    count_l = 0
    count_r = 0

    for k in 1:n

        plot!([xmin, xmid₁, xmid₂, xmax], [mean(gmins), mean(gmid₁s), mean(gmid₂s), mean(gmaxs)])
        gdiffall = abs(mean(gmid₁s) - mean(gmins)) + abs(mean(gdiffs)) + abs(mean(gmid₂s) - mean(gmaxs))
        @show gdiffall
        if (xmid₁ == xmid₂ == xmin) | (xmid₁ == xmid₂ == xmax)
            break
        elseif gdiffall < gtol
            @show "Breaking due to gdiffall"
            break
        end

        if mean(ci) > 0
            xmax = xmid₂
            xmid₂ = xmid₁
            h = invϕ * h
            xmid₁ = round(Int, xmin + invϕ² * h)
            count_l += 1
        else
            xmin = xmid₁
            xmid₁ = xmid₂
            h = invϕ * h
            xmid₂ = round(Int, xmin + invϕ * h)
            count_r += 1
        end

        gdiffs = zeros(0)
        gmins = zeros(0)
        gmid₁s = zeros(0)
        gmid₂s = zeros(0)
        gmaxs = zeros(0)
        for i ∈ 1:C
            gmin = g(xmin)
            gmid₂ = g(xmid₂)
            gmid₁ = g(xmid₁)
            gmax = g(xmax)
            append!(gmins, gmin)
            append!(gmid₁s, gmid₁)
            append!(gmid₂s, gmid₂)
            append!(gmaxs, gmax)
            append!(gall, [(xmin, gmin), (xmid₁, gmid₁), (xmid₂, gmid₂), (xmax, gmax)])

            gdiff = gmid₂ - gmid₁
            append!(gdiffs, gdiff)
        end
        ci = confint(OneSampleTTest(gdiffs))

    end

    display(p)

    return mean([xmin, xmid₁, xmid₂, xmax]), gall

end


metrics = ["met1", "met2"]
model_names = ["mod1", "mod2", "mod3"]
real_ns = [10, 20, 30, 40]
λs = [0.5, 1, 1.5]
iterations = 5

num_real_ns = length(real_ns)
num_λs = length(λs)
num_metrics = length(metrics)
num_models = length(model_names)

model_steps = num_metrics
real_n_steps = model_steps * num_models
λ_steps = real_n_steps * num_real_ns
iter_steps = λ_steps * num_λs
total_steps = iter_steps * iterations


for i in 1:total_steps

    iter = ceil(Int, i / iter_steps)
    iterᵢ = ((i - 1) % iter_steps) + 1
    λ = λs[ceil(Int, iterᵢ / λ_steps)]
    λᵢ = ((iterᵢ - 1) % λ_steps) + 1
    real_n = real_ns[ceil(Int, λᵢ / real_n_steps)]
    real_nᵢ = ((λᵢ - 1) % real_n_steps) + 1
    model = models[ceil(Int, real_nᵢ / model_steps)]
    modelᵢ = ((real_nᵢ - 1) % model_steps) + 1
    metric = metrics[modelᵢ]

    @show i, iterᵢ, λᵢ, real_nᵢ, modelᵢ
    @show iter, λ, real_n, model, metric
    println()

end
