import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:, 0]  # A single list of numbers/data
# print(x)
y = np.c_[data[:, 1]]  # matrix with one column y
# print(y)
# matrix (theta) with 2 columns (theta0 and theta1) | [[1 data]]
X = np.c_[np.ones(data.shape[0]), x]
# print(X)
m = X.shape[0]  # Number of training examples
n = X.shape[1]  # Number of features | 2 features
# print(m) number of rows |||||| 0 - Rows and 1 - Columns
# print(n) number of columns

iterations = 1500  # Number of iterations
alpha = 0.01  # Learning rate
# ***********Theta0 is b and Theta1 is m*************
# theta vector - 2 X 1 | y = mx + b | m and b are initialized to zero
theta = np.zeros((n, 1))  # column vector of Zeros
# print(theta)


def visualise_Data(x, y):
    plt.plot(x, y, 'bx')
    plt.show()


# print(X.dot(theta))
visualise_Data(x, y)  # Plotting the graph for the data


def costFunction(X, y, theta):
    # Direct multiplication of matrices X and theta (nx2 * 2*1 = n*1) |
    # h(theta) = theta0 + theta1*X | y = mx + b
    hypothesis = X.dot(theta)
    error = hypothesis - y
    sqrError = error**2  # Square Error
    J = np.sum(sqrError) / (2 * m)
    return J


def gradient_descent(X, y, theta):
    cost = []
    for i in range(iterations):
        hypothesis = X.dot(theta)
        theta = theta - ((alpha / m) * (X.T.dot(hypothesis - y)))
        J = costFunction(X, y, theta)
        cost.append(J)
    return theta, cost
print(cost)

weights = gradient_descent(X, y, theta)[0]


def plot_line(weights, X, y):
    plt.plot(x, y, 'bx')
    weights = y = np.c_[weights]
    equation = X.dot(weights)
    plt.plot(x, equation, '-')
    plt.show()


plot_line(weights, X, y)
