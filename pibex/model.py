import numpy as np
import pandas as pd

class Model():
    """Model class for the pibex package"""

    def __init__(self, f, nf, link, ilink, dist):
        """Constructor for the pibex model class"""
        self.f = f
        self.link = link
        self.dist = dist
        self.param = np.zeros(nf)

    def simulate(self, x):
        """pibex function for simulating data"""

        #compute design matrix
        F = x.apply(self.f,axis='index',result_type='expand')
        #dot design matrix with params
        FB = F.dot(self.param)
        #compute expected mu
        u = self.ilink(FB)

        return u


