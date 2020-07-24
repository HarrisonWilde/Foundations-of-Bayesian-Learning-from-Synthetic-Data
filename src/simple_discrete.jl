using Distributions
#using Plots


# think about prior for parameters- maybe also discrete!

r=5
p=0.1
n_real=20
n_realH=1000
n_eval=100
dgp = NegativeBinomial(r, p)
real_data = rand(dgp,n_real)
eval_data= rand(dgp,n_eval)
real_dataH = rand(dgp,n_realH)
syn_data =



function geom_mechanism(v,geom_p=0.5)
    res=v+rand(Geometric(geom_p))*(2*(rand()>0.5)-1)
    res*(res>0)+0 # hack
end
syn_data = geom_mechanism.(real_dataH)



#  Geometirc https://drive.google.com/file/d/114GWEJdlEsHa7wKtKFHEnpG56MVQenlq/view?usp=sharing

function lpmfnb(r::Float64,p::Float64,v::Array{Int64})
    logpdf(NegativeBinomial(r, p),v)
end


prior_params=[[rs,ps] for rs in [1:5;] for ps in [0.1,0.2,0.3]]
prior_weights= [1.0 for i=1:length(prior_params)]

logpdf(dgp, real_data)

function posterior(prior_weights,prior_params,data)
    loglikes=[sum(lpmfnb(pp...,data)) for pp in prior_params]
    loglikes=loglikes.-minimum(loglikes)
    posterior_v= prior_weights .* exp.(loglikes)
    posterior_v=posterior_v/sum(posterior_v)
    return posterior_v
end
