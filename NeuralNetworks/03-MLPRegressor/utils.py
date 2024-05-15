#%%
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_targets(targets, function_latex):
    plt.plot(targets)
    plt.ylabel('$Y$')
    plt.xlabel('$X$')
    plt.title(f"MLP Regressor of underlying function {function_latex}")
    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', "MLPRegressor_target.png"))
    plt.show()

def plot_results(targets, predictions, function_latex):
    plt.plot(targets, label = 'Target')
    plt.plot(np.concatenate(predictions), label = 'Predictions')
    plt.legend()
    plt.ylabel('$Y$')
    plt.xlabel('$X$')
    plt.title(f"MLP Regressor of {function_latex}")
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', "MLPRegressor_results.png"))
    plt.show()

def plot_MSE(error):
    plt.plot(error)
    plt.ylabel('MSE')
    plt.xlabel('Iterations')
    plt.title("MLP Regressor MSE")
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', "MLPRegressor_mse.png"))
    plt.show()
