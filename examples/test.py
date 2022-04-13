import nest_asyncio
nest_asyncio.apply()

import pibex as pbx
import numpy as np
import bambi as bm
import pandas as pd
import arviz as az
import patsy as pt
import formulae as fm
import stan as stan
import matplotlib.pyplot as plt

#%%

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
y = true_regression_line + rng.normal(scale=0.5, size=size)

data = pd.DataFrame(dict(x=x, y=y))

#%%

m = pbx.Model('y~x')

f=m.fit(data)

d=pbx.design(m)

m.simulate(d)

#%%


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="x", ylabel="y", title="Generated data and underlying model")
ax.plot(x, y, "x", label="sampled data")
ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
plt.legend(loc=0);

#%%

model = bm.Model("y ~ x", data)
trace = model.fit(draws=3000, cores=1)

#%%

tst_mod=pbx.Model("y ~ x")
tst_mod.fit(data)

#%%
az.plot_trace(trace, figsize=(10, 7));



#%%

df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5],
                   "y": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7]})

model = bm.Model("y ~ x", df)
fit = model.fit(draws=3000, cores=1)
az.plot_trace(fit, figsize=(10, 7));


#%%

df = pd.DataFrame({"x": [],
                   "y": []})

model = bm.Model("y ~ x", df)
fit = model.fit(draws=3000, cores=1)


#%%

mod = fm.model_description('y ~ x1*x2')

df = pd.DataFrame({"y": [0, 1, 2, 3, 4, 5],
                   "x1": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7],
                   "x2": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7],
                   "x1:x2": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7]})

mat = fm.design_matrices('y ~ x1*x2', df)

mat2 = fm.matrices.DesignMatrices(mod, df, 0)


#%%

# create a dataset
df = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5], "y": [0.2, 1.4, 2.5, 6.1, 8.9, 9.7]})
# plot the dataset
df.plot(x="x",y="y", kind="scatter", color="r", title="Dataset to analyse")

data_dict = {"x": df["x"].astype(float).to_numpy(), "y": df["y"].to_numpy(), "N": len(df)}

#%%

linear_code = """
data {
  int<lower=1> N;
  vector[N] y;
  vector[N] x;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  // priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5); 

  // model
  y ~ normal(x * beta + alpha, sigma);
}
"""

posterior = stan.build(linear_code, data=data_dict, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000)

#%%

stan_results = fit.to_frame()
print(stan_results.describe())

#%%

for row in range(0, len(stan_results)):
    fit_line = np.poly1d([stan_results["beta"][row], stan_results["alpha"][row]])
    x = np.arange(6)
    y = fit_line(x)
    plt.plot(x, y, "b-", alpha=0.025, zorder=1)
    
    
plt.scatter(df["x"], df["y"], c="r", zorder=2)
plt.title("All Fits Together")
plt.ylim([0, 12])
plt.ylabel("y")
plt.xlim([0, 5])
plt.xlabel("x")
plt.show()              