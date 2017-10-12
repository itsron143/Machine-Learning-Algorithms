import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Load Data
data = np.loadtxt('ex2data1.txt', delimiter=',')
x = data[:, 0:2]  # Training Data
y = np.c_[data[:, 2]]  # Column Vector of ones and zeros
# print(X)
# print(y)
X = np.c_[np.ones((x.shape[0], 1)), x]  # Adding theta0 columns of Ones
# print(X)
m = X.shape[0]  # Number of Training Examples
n = X.shape[1]  # Number of features
iterations = 1500  # Gradient Descent
alpha = 0.01
l = 1
theta = np.zeros((n, y.shape[1]))  # 3x1 theta0 is always 1
# print(theta)

# Visualise Data
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y)
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Sigmoid Function


def costFunction(X, y, theta):
    hypothesis = sigmoid(X.dot(theta))  # hypothesis
    one = np.ones(y.shape)  # Vector of ones for Matrix Subtraction
    J = (-1 / m) * np.sum(np.multiply(y, np.log(hypothesis)) +
                          np.multiply((one - y), np.log(one - hypothesis)))
    return J


print("Initial Cost is: ", costFunction(X, y, theta))


def gradientDescent(X, y, theta):  # Gradient Descent Algorithm
    for i in range(iterations):
        hyp = sigmoid(X.dot(theta))
        theta = theta - (alpha / m) * (X.T.dot(hyp - y))
    return theta


print("Weights from Gradient Descent are: \n", gradientDescent(X, y, theta))


def Predict(X, theta):
    hyp = sigmoid(X.dot(theta))
    if float(hyp) >= 0.5:
        print("True Prediction! Admitted! Accuracy - ", float(hyp) * 100)
    else:
        print("Sorry, Not admitted Accuracy - ", float(hyp) * 100)


x = np.array([1, 45, 85])
Predict(x, gradientDescent(X, y, theta))
