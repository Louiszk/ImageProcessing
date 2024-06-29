import tensorflow as tf
from keras._tf_keras.keras.datasets import mnist 

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

X_train = X_train / 255
X_test = X_test / 255


max_iters = [5, 10, 15]
learning_rates = [0.1, 0.01]
layers = [(50, 50), (50, 50, 50), (50, 50, 50, 50)]
solvers = ['adam', 'sgd', 'lbfgs']


results = []
for i, (max_iter, learning_rate, layer, solver) in enumerate(itertools.product(max_iters, learning_rates, layers, solvers)):

    print("Evaluating Combination ", i)
    mlp = MLPClassifier(hidden_layer_sizes=layer, max_iter=max_iter, learning_rate_init=learning_rate, solver=solver)
    mlp.fit(X_train, y_train)
    

    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store the results
    results.append({
        'max_iter': max_iter,
        'learning_rate': learning_rate,
        'layers': layer,
        'solver': solver,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })


results_df = pd.DataFrame(results)
results_df.to_csv('mlp_results.csv', index=False, sep=';')

