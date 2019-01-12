import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import  matplotlib
import seaborn as sns

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

# %config InlineBackend.figure_formats = {'pdf',}

sns.set_context('notebook')
sns.set_style('white')
def warmUpExercise():
    print(np.identity(5))

warmUpExercise()

data = np.loadtxt('C:\\Users\Ritesh\Desktop\machine-learning-ex1\ex1/ex1data1.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]
print(X)
print(y)

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    J = 0
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum(np.square(h - y))

    print (J)
computeCost(X,y)


def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
        J_history[iter] = computeCost(X, y, theta)
    return (theta, J_history)


# theta for minimized cost J
theta, Cost_J = gradientDescent(X, y)
print('theta: ', theta.ravel())

plt.plot(theta)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()

xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()