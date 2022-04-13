import numpy as np
import pandas as pd
#import pystan as pst
import bambi as bm


#%%
class Model:
     
    def __init__(self,formula,link="identity"):
        
        self.formula = formula
        
    def fit(self,data):

        self.mod = bm.Model(self.formula,data)
        self.idata = self.mod.fit(draws=3000, cores=1)

        return self.idata

    def simulate(self,data):

        sim_data = self.mod.predict(self.idata,kind='pps',data=data,inplace=False)

        return sim_data



