import pibex as pbx
import numpy as np
import pandas as pd


f = lambda x: [1, x, x**2]
nf=3
link = lambda x: np.log(x)
ilink = lambda x: np.exp(x)
dist = 'norm'

model = pbx.Model(f,nf,link,ilink,dist)


x=pd.DataFrame({'inputs':[-1,0,1]})


u = model.simulate(x)

print(u)