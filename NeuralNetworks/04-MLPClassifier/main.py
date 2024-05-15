#%%
from model import MLPClassifier
from utils import *
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

mnist = fetch_openml('mnist_784')
X, y = mnist.data / 255.0, mnist.target.astype(int)

X = np.array(X)
y = np.array(y).reshape(-1,1)

encoder = OneHotEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.toarray()
y_test = y_test.toarray()

model = MLPClassifier(hidden_sizes=[128,128], alpha=0.2)

error = model.train(X_train, y_train, batch = 128, epochs=25)

plot_error(error)

predictions = model.test(X_test)

y_test = encoder.inverse_transform(y_test)
X_test = X_test.reshape((14000, 28, 28))

print(f"confusion matrix: {confusion_matrix(y_test, predictions)}", '\n')
print(f"accuracy: {accuracy_score(y_test, predictions)}")

animate(X_test, y_test, predictions)

# %%
