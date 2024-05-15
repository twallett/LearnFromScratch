#%%
import numpy as np
from tqdm import tqdm

class MLPClassifier:
    def __init__(self, hidden_sizes, alpha):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.error = []
        self.predictions = []

    def logsig(self, n):
        return 1 / (1 + np.exp(-n))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, pred, true):
        return -np.mean(np.sum(true * np.log(pred), axis=1))
        
    def initialize_weights_and_biases(self, layers_size):
        self.weights = [np.random.randn(layers_size[i], layers_size[i + 1]) for i in range(len(layers_size) - 1)]
        self.biases = [np.zeros((1, layers_size[i + 1])) for i in range(len(layers_size) - 1)]

    def forward(self, inputs):
        activations = [inputs]
        for i in range(len(self.weights) - 1):
            n = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.logsig(n)
            activations.append(a)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = self.softmax(z_output)
        activations.append(a_output)
        return activations
    
    def backward(self, a_output, target_batch, batch, activations):
        gradients = [a_output - target_batch]
        for i in range(len(self.weights) - 2, -1, -1):
            derivative_n = gradients[-1].dot(self.weights[i+1].T) * (activations[i+1] * (1 - activations[i+1]))
            derivative_weight = np.dot(activations[i].T, derivative_n)
            derivative_bias = np.sum(derivative_n, axis=0)
            gradients.append(derivative_n)
            self.weights[i] -= (self.alpha * derivative_weight / batch)
            self.biases[i] -= (self.alpha * derivative_bias / batch)

    def train(self, inputs, targets, batch = 32, epochs = 100):
        
        input_size = inputs.shape[1]
        output_size = targets.shape[1]
        layer_sizes = [input_size] + self.hidden_sizes + [output_size]
        
        self.initialize_weights_and_biases(layer_sizes)
        
        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            for i in tqdm(range(0, len(inputs), batch)):
                
                # BATCH RANGE 
                
                input_batch = inputs[i:i+batch]
                target_batch = targets[i:i+batch]
                
                # FOWARD-PROPAGATION
                
                activations = self.forward(input_batch)
                a_output = activations[-1]

                # CROSS ENTROPY LOSS

                loss = self.cross_entropy_loss(a_output, target_batch)
                self.error.append(loss)

                # BACK-PROPAGATION 

                self.backward(a_output, target_batch, batch, activations)
            print(f"train loss: {loss.__round__(2)}")
            self.error.append(loss)

        return self.error

    def test(self, inputs):
        
        for i in range(len(inputs)):

            # FOWARD-PROPAGATION

            input_batch = inputs[i:i+1].reshape((1,inputs.shape[1]))
            
            activations = self.forward(input_batch)
            a_output = activations[-1]

            self.predictions.append(np.argmax(a_output))
            
        return self.predictions