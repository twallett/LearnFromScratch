#%%
import numpy as np 
import sympy as sp

class LinearMinimization:
    def __init__(self):
        self.x_k = []
        
    def forward(self, function, initial_condition, n_iter = 1000, threshold = 1e-04):
        
        x, y = sp.symbols('x y', real=True)
        
        hessian_sym = sp.hessian(function, [x,y])
        hessian  = np.array(hessian_sym, dtype= int)
        
        for i in range(n_iter):
        
            gradient = np.array(([sp.diff(function, x).replace(x, initial_condition[0][0]).replace(y, initial_condition[1][0])], 
                                [sp.diff(function, y).replace(x, initial_condition[0][0]).replace(y, initial_condition[1][0])]), 
                                dtype=float).reshape(2,1)
            
            p = -1 * gradient
            
            alpha = -1 * (gradient.T @ p)/(p.T @ (hessian @ p))
            
            x_k = initial_condition - (alpha * gradient)
            
            initial_condition = x_k 
            
            self.x_k.append(x_k)
            
            if np.abs(np.linalg.norm(gradient)) < threshold:
                break
        
        return np.array(self.x_k)