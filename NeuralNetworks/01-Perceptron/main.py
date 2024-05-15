#%%
from model import Perpceptron
from utils import *
import numpy as np

EPOCHS = 2
HIDDEN = 2

inputs = np.array([[-2,2], [-1,2], [-2,1], [-1,1], 
                   [1,1], [1,2], [2,1], [2,2], 
                   [1,-1], [1,-2], [2,-1], [2,-2], 
                   [-2,-1], [-2,-2], [-1,-2], [-1,-1]], 
                  dtype=float)

targets = np.array([0,0,0,0,0,0,0,0,
                    1,1,1,1,1,1,1,1], 
                   dtype=float)
targets_xor = np.array([0,0,0,0,1,1,1,1,
                        0,0,0,0,1,1,1,1],
                       dtype=float)

model = Perpceptron(hidden=HIDDEN)
model_xor = Perpceptron(hidden=HIDDEN)

model.reset_param()
model_xor.reset_param()

for epoch in range(EPOCHS):
    error = model.train(inputs = inputs, targets=targets)
    error_xor = model_xor.train(inputs = inputs, targets=targets_xor)
    
plot_sse(error)
plot_sse(error_xor, xor=True)

parameters = model.get_param() 
parameters_xor = model_xor.get_param() 

animate(inputs, targets, parameters)
animate(inputs, targets_xor, parameters_xor, xor = True)
    
#%%
