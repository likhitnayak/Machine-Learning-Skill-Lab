import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
np.random.seed(39) # reproducibility

# -------------- user-adjustable hyper-parameters -----------------

n_train = 48 # points in training set
n_test = 12 # points in test set
noise_sigma = 0.25 # std-dev of Gaussian noise added to targets
degree_range = range(1, 10) # polynomial degrees to evaluate
n_runs = 200 # independent training sets
x_test_grid = np.linspace(0, 2*np.pi, 10).reshape(-1, 1) # fixed and unseen test inputs
true_fx = np.sin(x_test_grid).ravel() # ground-truth
rng = np.random.default_rng(42)
# -----------------------------------------------------------------

# Generate noisy sine-wave data

X = np.linspace(0, 6*np.pi, n_train + n_test).reshape(-1, 1)
y_true = np.sin(X).ravel()
y_noisy = y_true + np.random.normal(0, noise_sigma, size=y_true.shape)
# print(X.shape, y_true.shape, y_noisy.shape)

# Split into training and test subsets  

X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=n_test, shuffle=False) # Can also be y_true if we want to use noise-free data
train_mse, test_mse = [], []

for degree in degree_range:
    # Train model
    poly = PolynomialFeatures(degree, include_bias=False)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train), y_train)
    # Evaluate model
    y_train_pred = model.predict(poly.transform(X_train))
    y_test_pred = model.predict(poly.transform(X_test))
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(degree_range, train_mse, label='Training MSE')
plt.plot(degree_range, test_mse, label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Compute bias^2 and variance
bias2_all, var_all, noise_all = [], [], []
for degree in degree_range:
    predictions = []
    for  run in range(n_runs):
        # ---- draw a fresh training sample -----------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=n_test, shuffle=False)
        # ---- train a model -----------------------------
        poly = PolynomialFeatures(degree, include_bias=False)
        model = LinearRegression()
        model.fit(poly.fit_transform(X_train), y_train)
        # ---- evaluate the model on the test grid -----------------------------
        y_test_pred = model.predict(poly.transform(x_test_grid))
        predictions.append(y_test_pred)
    predictions = np.array(predictions)
    bias = np.mean(predictions, axis=0) - true_fx
    variance = np.var(predictions, axis=0)
    bias2_all.append(bias**2)
    var_all.append(variance)
    noise_all.append(noise_sigma**2)

bias2_all = np.array(bias2_all)
var_all = np.array(var_all)
noise_all = np.array(noise_all)
generalization_error = bias2_all + var_all + noise_all[:, np.newaxis]

plt.figure(figsize=(10, 6))
# plt.plot(degree_range, bias2_all, label='Bias^2')
# plt.plot(degree_range, var_all, label='Variance')
# plt.plot(degree_range, noise_all, label='Noise')
plt.plot(degree_range, generalization_error, label='Generalization Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('Generalization Error')
plt.show()