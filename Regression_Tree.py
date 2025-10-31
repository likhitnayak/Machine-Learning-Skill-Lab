from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Boston Housing Dataset.
from sklearn.datasets import fetch_openml
boston = fetch_openml(name='boston', as_frame=True, parser='auto')
# The data is in boston.data, and the target (house prices) is in boston.target.
X = boston.data
y = boston.target

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Fit a DecisionTreeRegressor and tune the 'max_depth' hyperparameter.
depths = range(1, 11) # The range of max_depth values to test
regressor_models = []
test_mses = []
print("\n--- Starting Hyperparameter Tuning ---")
for depth in depths:
    # Create and train the model on training set
    regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_test_pred = regressor.predict(X_test)

    # Calculate the MSE on the test set
    mse = mean_squared_error(y_test, y_test_pred)
    test_mses.append(mse)
    print(f"max_depth = {depth}, Test MSE = {mse:.4f}")
    regressor_models.append(regressor)

# 4. Get the Best Model
best_depth_index = np.argmin(test_mses)
optimal_depth = depths[best_depth_index]
best_regressor = regressor_models[best_depth_index]
print(f"\nOptimal max_depth found: {optimal_depth} (with lowest test MSE)")


# 5. Plot the Final Tree
plt.figure(figsize=(10, 6)) 
# For regression, the 'value' in each node is the predicted average value for that node.
# The 'filled=True' argument colors the nodes based on their value.
plot_tree(best_regressor,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Boston Housing Prices", fontsize=16)
plt.show()

# 6. Evaluate the Best Model on the Test Set
y_pred = best_regressor.predict(X_test)
# Calculate R-squared and Mean Squared Error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Performance on Test Set:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")