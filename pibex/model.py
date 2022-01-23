import numpy as np
import pandas as pd

class Model():
    """Model class for the pibex package"""

    def __init__(self, f, link, dist):
        """Constructor for the pibex model class"""
        self.f = f
        self.link = link
        self.dist = dist

    def simulate(self, x):
        """pibex function for simulating data"""

        u=x.apply(self.f,'index')


