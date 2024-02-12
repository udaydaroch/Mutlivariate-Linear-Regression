import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('home.txt', names=["size", "bedroom", "price"])

# Standardize features
data = (data - data.mean()) / data.std()

# Extract features and target
X = data.iloc[:, :-1]
X.insert(0, 'Ones', 1)  # Add intercept term
y = data.iloc[:, -1]

# Gradient Descent
def compute_cost(X, y, theta):
    return np.sum(np.square(X @ theta.T - y)) / (2 * len(X))

def gradient_descent(X, y, theta, alpha, iters):
    cost_history = []
    for _ in range(iters):
        theta = theta - (alpha / len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

theta = np.zeros(X.shape[1])  # Initialize parameters
alpha = 0.01
iters = 1000

# Train model
theta, cost_history = gradient_descent(X.values, y.values, theta, alpha, iters)
final_cost = compute_cost(X.values, y.values, theta)

# Plot cost over iterations
plt.plot(np.arange(iters), cost_history, 'r')  
plt.xlabel('Iterations')  
plt.ylabel('Cost')  
plt.title('Error vs. Training Epoch')
plt.show()

print("Optimized parameters:", theta)
print("Final cost:", final_cost)
