import numpy  as np
import itertools
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------
# 1. Load & prepare the data
# -----------------------------------------------------------------
boston = fetch_openml(name="Boston", version=1, as_frame=True)
X_full = boston.data.astype(float)
y = boston.target.astype(float)

n_tot, p_tot = X_full.shape
print(f"Data set: {n_tot} observations, {p_tot} candidate predictors\n")

# -----------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------
def bic_linear_regression(X, y):
    """
    Calculates the Bayesian Information Criterion (BIC) for a linear regression model.
    """
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the predictions and calculate the residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Calculate the Residual Sum of Squares (RSS)
    rss = np.sum(residuals**2)
    
    # Calculate the BIC
    # The number of parameters is p (number of features) + 1 (for the intercept).
    # The formula used is: n * log(RSS / n) + (p + 1) * log(n)
    n, p = X.shape
    bic = n * np.log(rss / n) + (p + 1) * np.log(n)

    return bic

def cv_mse_linear_regression(X, y, kfolds, random_state=1):
    """
    Calculates the cross-validated Mean Squared Error (MSE) for a linear regression model.
    """
    model = LinearRegression()

    # Scikit-learn's cross-validation tools are designed to maximize a score
    # As a result, the error terms are negated so that a lower error (which is better) becomes a higher score.
    cv = KFold(n_splits=kfolds, shuffle=True, random_state=random_state)
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')

    # We return the negative of the mean score to get the positive MSE.
    return -np.mean(mse_scores)

# -----------------------------------------------------------------
# 3. Best single predictor
# -----------------------------------------------------------------
predictors_name = list(X_full.columns)
bic_candidates = []
cv_mse_candidates = []
for predictor in predictors_name:
    X_candidate = X_full[[predictor]]
    bic_val = bic_linear_regression(X_candidate, y)
    cv_val = cv_mse_linear_regression(X_candidate, y, 11)
    bic_candidates.append((bic_val, predictor))
    cv_mse_candidates.append((cv_val, predictor))
    # print(predictor, bic_val, cv_val)

# Which predictor has the lowest BIC and which has the lowest CV-MSE?
bic_candidates.sort(key=lambda t: t[0])
cv_mse_candidates.sort(key=lambda t: t[0])
print(bic_candidates[0])
print(cv_mse_candidates[0])


# -----------------------------------------------------------------------------------------------------
# 4. Best predictor combination, considering a greedy search of all possible combinations of predictors
# -----------------------------------------------------------------------------------------------------

all_predictors = list(X_full.columns)
all_bic_candidates = []
all_cv_mse_candidates = []

for i in range(p_tot):
    bic_candidates = []
    cv_mse_candidates = []
    for combo in itertools.combinations(all_predictors, i+1):
        X_combo = X_full[list(combo)]
        bic_val = bic_linear_regression(X_combo, y)
        cv_val = cv_mse_linear_regression(X_combo, y, 11)
        bic_candidates.append((bic_val, combo))
        cv_mse_candidates.append((cv_val, combo)) 
    bic_candidates.sort(key=lambda t: t[0])
    cv_mse_candidates.sort(key=lambda t: t[0])
    print("Best BIC for ", i+1, " predictors: ", bic_candidates[0]) # prints the best BIC for a given number of predictors
    print("Best CV-MSE for ", i+1, " predictors: ", cv_mse_candidates[0]) # prints the best CV-MSE for a given number of predictors
    all_bic_candidates.extend(bic_candidates)
    all_cv_mse_candidates.extend(cv_mse_candidates)

print("Greedy search finished.")
all_bic_candidates.sort(key=lambda t: t[0])
all_cv_mse_candidates.sort(key=lambda t: t[0])
print("Best BIC: ", all_bic_candidates[0])
print("Best CV-MSE: ", all_cv_mse_candidates[0])
