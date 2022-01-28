from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)
# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
##alpha = 1.0
alpha = 0.1 #modified
iterations = 100

# plot predictions for every iteration?
##do_plot = True
do_plot = False

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print(f"Theta values after optimization: {theta_final}")

#########################################
# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples
X1 = np.array([1650, 3])
X1_normalized = (X1 - mean_vec) / std_vec
X1_normalized = np.append(np.ones((1, 1)), X1_normalized, axis=1)
predicted_price_1 = calculate_hypothesis(X1_normalized, theta_final, 0)
print("Predicted price for house with {} sq. ft. and {} bedrooms is {:.2f}".format(X1[0], X1[1], predicted_price_1))

X2 = np.array([3000, 4])
X2_normalized = (X2 - mean_vec) / std_vec
X2_normalized = np.append(np.ones((1, 1)), X2_normalized, axis=1)
predicted_price_2 = calculate_hypothesis(X2_normalized, theta_final, 0)
print("Predicted price for house with {} sq. ft. and {} bedrooms is {:.2f}".format(X2[0], X2[1], predicted_price_2))
########################################/
