import numpy as np

class Perpceptron:
    def __init__(self, hidden):
        self.hidden = hidden
        self.W = np.zeros((1,self.hidden), dtype=float)
        self.b = 0
        self.error = []
        self.param = []

    def hardlim(self, n):
        return 0 if n < 0 else 1
    
    def train(self, inputs, targets):
        for i in range(len(inputs)):
            error = targets[i] - self.hardlim( np.dot(self.W, inputs[i]) + self.b ) # a = F( W*p + b )
            self.W += (error * inputs[i])
            self.b += error
            self.error.append(error)
            self.param.append([self.W.copy(), self.b])
        return np.array(self.error)
    
    def get_param(self):
        return self.param
    
    def reset_param(self):
        self.W = np.zeros((1,self.hidden), dtype=float)
        self.b = 0
        self.error = []
        self.param = []