#%%
import numpy as np
import sympy as sp
from numpy import linalg as LA

class NewtonsMethod:
    def __init__(self):
        self.x_k = []

    def forward(self, function, initial_condition):
        
        self.x_k.append(initial_condition)
        
        x, y = sp.symbols('x y', real=True)
        
        hessian_sym = sp.hessian(function, [x,y])
        hessian  = np.array(hessian_sym, dtype= int)
        
        gradient = np.array([sp.diff(function, x).replace(x,initial_condition[0][0]).replace(y,initial_condition[1][0]), 
                             sp.diff(function, y).replace(x,initial_condition[0][0]).replace(y,initial_condition[1][0])], 
                            dtype=float).reshape(2,1)
        
        x_k = initial_condition - (LA.inv(hessian) @ gradient)
            
        self.x_k.append(x_k)
        
        return np.array(self.x_k)