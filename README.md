Multivariate Linear Regression using Gradient Descent

This repository contains a Python script implementing multivariate linear regression using gradient descent optimization. The script is designed to analyze housing data with features such as size and number of bedrooms to predict the price of a house.

Key Components:

Data Loading and Preprocessing:
The script loads the housing data from a CSV file and preprocesses it by standardizing the features using z-score normalization.

Gradient Descent Optimization:
It defines a gradient descent function to iteratively update model parameters to minimize the mean squared error (MSE) cost function.

Cost Function Calculation:
The cost function calculates the MSE between predicted and actual house prices to measure the model's performance.

Hyperparameters Setting:
Hyperparameters such as learning rate and number of iterations are set to control the optimization process.

Results Visualization:
The script visualizes the convergence behavior of the gradient descent algorithm by plotting the cost function over the iterations.
Usage:

To use the script, ensure that Python and the required libraries (NumPy, Pandas, and Matplotlib) are installed.
Update the file path to your housing data in the read_csv() function.
Adjust hyperparameters such as learning rate and number of iterations as needed.
Run the script and observe the optimized model parameters and final cost.
References:

For more details on multivariate linear regression and gradient descent, refer to relevant literature on machine learning and optimization.
Author:
Tan Moy (https://medium.com/we-are-orb/multivariate-linear-regression-in-python-without-scikit-learn-7091b1d45905)
Uday Daroch

