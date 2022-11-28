
# How does the naive Bayes classifier compare to neural networks and decision trees on perceptual data
# Training all three on the MNIST data fand Rank the them in terms of accuracy and running time.
# 

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import random
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# divide data into test set and trainning set
train_sample = 5000

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])

X = X.to_numpy()    # fix error X being not treated as a numpy array
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_sample, test_size=10000)

rank = []

def train_and_test(clf):
    import time
    start = time.time()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    end = time.time() - start
    del time

    rank.append((clf.__class__.__name__, score, end))

def main():
    clf = MLPClassifier(hidden_layer_sizes=250, max_iter=10000)
    train_and_test(clf)

    clf = tree.DecisionTreeClassifier(max_leaf_nodes=250)
    train_and_test(clf)

    clf = MultinomialNB()
    train_and_test(clf)

    print(rank)

if __name__ == '__main__':
    main()

