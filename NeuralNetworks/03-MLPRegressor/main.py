#%%
from model import MLPRegressor
from utils import *
import numpy as np
import math

HIDDEN = [20,20]
ALPHA = 0.2

def f(x):
    return 1 + np.sin((math.pi/4) * x) 

function_latex = "$f(x) = 1 + sin(\\frac{\pi}{4}x)$"
inputs = np.linspace(-2,2).reshape(-1,1)
targets = f(np.linspace(-2,2)).reshape(-1,1)

plot_targets(targets, function_latex)

model = MLPRegressor(hidden_sizes=HIDDEN, alpha=ALPHA)

error = model.train(inputs, targets)

plot_MSE(error)

predictions = model.test(inputs, targets)

plot_results(targets, predictions, function_latex)

# %%
