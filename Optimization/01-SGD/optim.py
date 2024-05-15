#%%
import numpy as np
import sympy as sp

class SGD:
    def __init__(self, alpha):
        self.alpha = alpha
        self.x_k = []
        
    def forward(self, function, initial_condition, n_iter = 1000, threshold = 1e-4):
        
        x, y = sp.symbols('x y', real=True)
        
        gradient = np.array([sp.diff(function, x).replace(x,0).replace(y,0),
                             sp.diff(function, y).replace(x,0).replace(y,0)],
                            dtype=float).reshape(2,1)

        for _ in range(n_iter):
            
            x_k = initial_condition - (self.alpha * gradient)
            
            gradient = np.array(([sp.diff(function, x).replace(x, x_k[0][0]).replace(y, x_k[1][0])], 
                                [sp.diff(function, y).replace(x, x_k[0][0]).replace(y, x_k[1][0])]), 
                                dtype=float).reshape(2,1)
            
            initial_condition = x_k
            
            self.x_k.append(x_k)
            
            if np.abs(np.linalg.norm(gradient)) < threshold:
                break
        
        return np.array(self.x_k)