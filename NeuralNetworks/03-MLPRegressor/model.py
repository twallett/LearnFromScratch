#%% 
import numpy as np
from tqdm import tqdm

class MLPRegressor:
    def __init__(self, hidden_sizes, alpha):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.error = []
        self.predictions = []
        
    def initialize_weights_and_biases(self, layer_sizes):
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        
    def logsig(self, n):
        return 1 / (1 + np.exp(-n))

    def forward(self, inputs):
        activations = [inputs]
        for i in range(len(self.weights) - 1):
            n = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.logsig(n)
            activations.append(a)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = z_output
        activations.append(a_output)
        return activations
    
    def backward(self, target_batch, pred_batch, batch, activations):
        gradients = [-2 * (target_batch - pred_batch)]
        for i in range(len(self.weights) - 2, -1, -1):
            derivative_n = gradients[-1].dot(self.weights[i+1].T) * activations[i+1] * (1 - activations[i+1])
            derivative_weight = np.dot(activations[i].T, derivative_n)
            derivative_bias = np.sum(derivative_n, axis= 0)
            gradients.append(derivative_n)
            self.weights[i] -= (self.alpha * derivative_weight / batch)
            self.biases[i] -= (self.alpha * derivative_bias / batch)

    def train(self, inputs, targets, batch = 32, epochs = 100):

        layer_sizes = [inputs.shape[1]] + self.hidden_sizes + [targets.shape[1]]
        
        self.initialize_weights_and_biases(layer_sizes)

        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            for i in tqdm(range(0, len(inputs), batch)):
                
                # BATCH RANGE 
                     
                input_batch = inputs[i:i + batch]
                target_batch = targets[i:i + batch]
                
                # FORWARD PROPAGATION 

                activations = self.forward(input_batch)
                pred_batch = activations[-1]
            
                # LOSS CALCULATION
                
                loss = np.mean((target_batch - pred_batch) ** 2)
                
                # BACK PROPAGATION AND WEIGHT UPDATES
                
                self.backward(target_batch, pred_batch, batch, activations)
            print(f"train loss: {loss.__round__(2)}")
            self.error.append(loss)
                
        return np.array(self.error)
    
    def test(self, inputs, targets, batch = 32):
        
        for i in range(0, len(inputs), batch):
            
                # BATCH RANGE 
                     
                input_batch = inputs[i:i + batch]
                target_batch = targets[i:i + batch]
                
                # FORWARD PROPAGATION 

                activations = self.forward(input_batch)
                pred_batch = activations[-1]
                
                self.predictions.append(pred_batch)
            
                # LOSS CALCULATION
                
                error = np.mean((target_batch - pred_batch) ** 2)
                
        return self.predictions 
                
                