import numpy as np

class ADALINE:
    def __init__(self, hidden, alpha):
        self.hidden = hidden
        self.alpha = alpha
        self.W = np.identity(self.hidden, dtype=float)
        self.b = 0
        self.error = []
        self.param = []

    def train(self, inputs, targets):
        for i in range(1, len(inputs) + 1):
            error = targets[:, i - 1:i] - (np.dot(self.W, inputs[:, i - 1:i]) + self.b)
            self.W += (2 * self.alpha * error @ inputs[:, i - 1:i].T)
            self.b += (2 * self.alpha * error)
            self.error.append(error)
            self.param.append([self.W.copy(), self.b])
            if error[0] == 0 and error[1] == 0:
                break
            else:
                continue
        return np.concatenate(self.error)
    
    def get_param(self):
        return self.param
    
    def reset_param(self):
        self.W = np.identity(self.hidden, dtype=float)
        self.b = 0
        self.error = []
        self.param = []