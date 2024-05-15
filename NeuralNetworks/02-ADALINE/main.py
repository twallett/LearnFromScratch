#%%
from model import ADALINE
from utils import *
import numpy as np

EPOCHS = 100
HIDDEN = 2
ALPHA = 0.003

inputs = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
                   [1, 2, -1, 0, 2, 1, -1, -2]])

targets = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
                    [-1, -1, 1, 1, -1, -1, 1, 1]])

model = ADALINE(hidden=HIDDEN, alpha = ALPHA)

model.reset_param()

for epoch in range(EPOCHS):
    error = model.train(inputs, targets)
    
plot_sse(error)

parameters = model.get_param()

animate(inputs, targets, parameters)

# %%
