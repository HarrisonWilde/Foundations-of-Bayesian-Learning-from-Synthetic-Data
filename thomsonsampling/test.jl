# using Turing, Distributions
# using KernelFunctions, Plots
# using PDMat
# using Statistics

# function g(x)
#     0.5*abs(x-25)^1.5
# end
# xend=50
# x=reshape([1:xend;],(xend,1))
# k=transform(GaussianKernel(),0.1)
# KM = kernelmatrix(k,x,obsdim=1)
# eigen(KM+eps()*100*I)
# plot(heatmap.([KM    ],yflip=true,colorbar=false)...,title=["K₁" "K" ])
# # PDMat(KM)
# # using PDMats
# # confused why not positive definite
# nm=MvNormal(KM+eps()*100*I)
# LinearAlgebra.ldltfact(KM)
# using LinearAlgebra
# # Use Turing to sample from posterior
# @model gdemo(x, y) = begin
#   global xend
#   s ~ MvNormal(Matrix(1.0I, xend, xend)) # prior
#   for i=1:length(x)
#   y[i] ~ Normal(s[x[i]], 1) # error dist
#   end
# end

# # initial samples
# ss=20
# xv=sample([1:50;],ss,replace=true)
# y=g.(xv)+randn(ss)
# #  Run sampler, collect results
# # https://discourse.julialang.org/t/multivariate-normal-with-positive-semi-definite-covariance-matrix/3029
# for i=1:10

# chn = sample(gdemo(xv, y), HMC(0.1, 5), 1000)
# ChainValues=chn[:s].value.data[:,:,1]

# # why is 59 dimensional
# chn.value
# # optimize to perform thomson sampling

# m=findmax(chn.values[end,:])[2] of last sample
# xv= vcat(xv,[m for i=1:10])
# y= vcat(y, [g(m)+randn() for i=1:10])

# ChainValues=chn[:s].value.data[:,:,1]
# fmean=mean(ChainValues,dims=[1])[1,:]
# fupper=map(i->Statistics.quantile(ChainValues[:,i],0.90,),[1:xend;])
# flower=map(i->Statistics.quantile(ChainValues[:,i],0.10,),[1:xend;])
# plot([1:xend;],fupper
# plot!([1:xend;],fmean)
# plot!([1:xend;],flower)
# savefig("step$i.png")
# end

# mean
# Try Michael's approach based on basis functions
# sample
# f(x)= sum a_i f_i
