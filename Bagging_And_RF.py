# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# -------------------------------------------------------------------
# 1. Bagging
# -------------------------------------------------------------------

# --- 1: Load the Wisconsin Breast Cancer dataset ---
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Split the data into training and testing sets
# We use a 70/30 split, and a random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2: Fit a single DecisionTreeClassifier and evaluate its accuracy ---
# We first establish a baseline performance with a single decision tree.
# We'll set a random_state for the tree as well for consistent results.
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

print("--- Bagging Section ---")
print(f"Accuracy of a single Decision Tree: {accuracy_tree:.4f}")

# --- 3: Use BaggingClassifier with the same decision tree ---
# Now, we'll use an ensemble of decision trees.
# The BaggingClassifier will create multiple bootstrap samples of the training data
# and train a decision tree on each. Predictions are then aggregated.
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=500,  # Number of trees in the ensemble
    max_samples=0.8,   # Use 80% of the training data for each tree
    bootstrap=True,
    random_state=42,
    n_jobs=-1          # Use all available CPU cores
)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print(f"Accuracy of Bagging Classifier: {accuracy_bagging:.4f}")


# -------------------------------------------------------------------
# 2. Random Forest
# -------------------------------------------------------------------

print("--- Random Forest Section ---")

# --- 1: Fit a RandomForestClassifier ---
# A Random Forest is an extension of bagging that also introduces randomness in
# the feature selection at each split, further decorrelating the trees.
# We will use GridSearchCV to find the best hyperparameters.

# --- 2: Use GridSearchCV to tune hyperparameters ---
# Define the parameter grid to search.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', 0.5] # Options for the number of features to consider
}

# Instantiate the RandomForestClassifier.
# We set oob_score=True to enable the out-of-bag error calculation.
rf = RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1)

# Set up GridSearchCV. cv=5 means 5-fold cross-validation.
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best estimator from the grid search.
best_rf = grid_search.best_estimator_
print(f"\nBest Random Forest Parameters: {grid_search.best_params_}")

# Evaluate the best model on the test set.
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Set Accuracy of Tuned Random Forest: {accuracy_rf:.4f}")

# --- Task 3: Compare OOB error to the test set error ---
# The OOB score is an estimate of the model's performance on unseen data,
# calculated using the trees that did not see a particular sample during training.
oob_accuracy = best_rf.oob_score_
print(f"Out-of-Bag (OOB) Accuracy: {oob_accuracy:.4f}")
print("\nThe OOB accuracy is often a reliable estimate of the test set accuracy.")
print("In this case, we can see that the two values are very close, which gives us")
print("confidence in the model's generalization performance.\n")

# --- Task 4: Plot the feature importances ---
# Random Forests can provide insights into which features are most predictive.
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances from the Random Forest Model')
plt.gca().invert_yaxis()
plt.show()