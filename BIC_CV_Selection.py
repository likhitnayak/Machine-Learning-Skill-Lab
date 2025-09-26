import numpy  as np
import matplotlib.pyplot as plt
import itertools
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm

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
# 3. Forward stepwise selection
# -----------------------------------------------------------------
remaining_features_bic = list(X_full.columns)
remaining_features_cv = list(X_full.columns)
selected_features_bic = [] # variables already in the model
selected_features_cv = [] # variables already in the model
bic_list_forward = []
cv_mse_list_forward = []
bic_path = []
cv_path = []

for i in range(p_tot):
    # ---- try adding each remaining predictor --------------------
    bic_candidates    = []
    cv_mse_candidates = []
    
    for candidate in remaining_features_bic:
        trial_vars = selected_features_bic + [candidate]
        X_trial = X_full[trial_vars]
        
        bic_val = bic_linear_regression(X_trial, y)
        bic_candidates.append((bic_val, candidate) )

    for candidate in remaining_features_cv:
        trial_vars = selected_features_cv + [candidate]
        X_trial = X_full[trial_vars]
        
        cv_val = cv_mse_linear_regression(X_trial, y, 11)
        cv_mse_candidates.append((cv_val, candidate))

    # ---- pick the variable that gives the lowest BIC ------------
    bic_candidates.sort(key=lambda t: t[0])
    best_bic, best_bic_var = bic_candidates[0]
    
    # ---- pick the variable that gives the lowest CV-MSE ------------
    cv_mse_candidates.sort(key=lambda t: t[0])
    best_cv_mse, best_cv_var = cv_mse_candidates[0]
    
    # -------------------------------------------------------------
    # store the results *after* adding the best variable
    # -------------------------------------------------------------
    selected_features_bic.append(best_bic_var)
    remaining_features_bic.remove(best_bic_var)
    bic_path.append(selected_features_bic.copy())
    selected_features_cv.append(best_cv_var)
    remaining_features_cv.remove(best_cv_var)
    cv_path.append(selected_features_cv.copy())

    bic_list_forward.append(best_bic)
    cv_mse_list_forward.append(best_cv_mse)

print("\nForward search finished.")

# -----------------------------------------------------------------
# 4. Which model is forward selected by …
#    (a) the minimum BIC on the training data?
#    (b) the minimum CV-MSE?
# -----------------------------------------------------------------
idx_bic = np.argmin(bic_list_forward)
idx_cv = np.argmin(cv_mse_list_forward)

print("\n-----------------------------------------------------------")
print("Model forward selected by minimum training BIC")
print("  Size :", idx_bic+1)
print("  Vars :", bic_path[idx_bic])
print("  BIC  :", bic_list_forward[idx_bic])

print("\nModel forward selected by minimum CV-MSE")
print("  Size :", idx_cv+1)
print("  Vars :", cv_path[idx_cv])
print("  CV-MSE :", cv_mse_list_forward[idx_cv])
print("-----------------------------------------------------------")

# -----------------------------------------------------------------
# 5. Backward stepwise selection
# -----------------------------------------------------------------

selected_features_bic = list(X_full.columns) # variables already in the model
selected_features_cv = list(X_full.columns) # variables already in the model
bic_list_backward = []
cv_mse_list_backward = []
bic_path = []
cv_path = []

while len(selected_features_bic) > 1:
    # ---- try removing each selected predictor --------------------
    bic_candidates    = []
    cv_mse_candidates = []
    
    for candidate in selected_features_bic:
        trial_vars = [var for var in selected_features_bic if var != candidate]
        X_trial = X_full[trial_vars]
        
        bic_val = bic_linear_regression(X_trial, y)
        bic_candidates.append((bic_val, candidate))
        
    for candidate in selected_features_cv:
        trial_vars = [var for var in selected_features_cv if var != candidate]
        X_trial = X_full[trial_vars]
        
        cv_val = cv_mse_linear_regression(X_trial, y, 11)
        cv_mse_candidates.append((cv_val, candidate) )

    # ---- pick the variable that gives the lowest BIC ------------
    bic_candidates.sort(key=lambda t: t[0])
    best_bic, best_bic_var = bic_candidates[0]
    
    # ---- pick the variable that gives the lowest CV-MSE ------------
    cv_mse_candidates.sort(key=lambda t: t[0])
    best_cv_mse, best_cv_var = cv_mse_candidates[0]
    # print(best_bic, best_var)
    
    # -------------------------------------------------------------
    # store the results *after* removing the best variable
    # -------------------------------------------------------------
    selected_features_bic.remove(best_bic_var)
    selected_features_cv.remove(best_cv_var)
    bic_path.append(selected_features_bic.copy())
    cv_path.append(selected_features_cv.copy())
    bic_list_backward.append(best_bic)
    cv_mse_list_backward.append(best_cv_mse)

print("\nBackward search finished.")


# -----------------------------------------------------------------
# 6. Which model is backward selected by …
#    (a) the minimum BIC on the training data?
#    (b) the minimum CV-MSE?
# -----------------------------------------------------------------
idx_bic = np.argmin(bic_list_backward)
idx_cv = np.argmin(cv_mse_list_backward)

print("\n-----------------------------------------------------------")
print("Model backward selected by minimum training BIC")
print("  Size :", idx_bic+1)
print("  Vars :", bic_path[idx_bic])
print("  BIC  :", bic_list_backward[idx_bic])

print("\nModel backward selected by minimum CV-MSE")
print("  Size :", idx_cv+1)
print("  Vars :", cv_path[idx_cv])
print("  CV-MSE :", cv_mse_list_backward[idx_cv])
print("-----------------------------------------------------------")

