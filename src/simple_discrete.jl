using Distributions
using Plots


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

#  Geometirc https://drive.google.com/file/d/114GWEJdlEsHa7wKtKFHEnpG56MVQenlq/view?usp=sharing


prior_params=[[rs,ps] for rs in [1:5;] for ps in [0.1,0.2,0.3]]
prior_weights= [1.0 for i=1:length(prior_params)]

logpdf(dgp, real_data)
posterior=



pdf(NegativeBinomial(prior_params[1]...),1000)
