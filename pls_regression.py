from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
import typing


def train_model(X_train, y_train, upper_bound=21):
    # Create a PLS model
    pls = PLSRegression()

    # Use GridSearchCV to find the best number of components using KFold cross-validation
    param_grid = {"n_components": list(range(1, upper_bound))}
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(pls, param_grid, cv=kf, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    # Best number of components
    best_n_components = grid_search.best_params_["n_components"]
    print(f"Best number of components: {best_n_components}")

    # Fit the PLS model with the best number of components
    pls_best = PLSRegression(n_components=best_n_components)
    pls_best.fit(X_train, y_train)

    return pls_best

# Calculate VIP (Variable Importance in Projection) scores for PLS
def calculate_vip(pls: PLSRegression, X: np.ndarray[typing.Any, typing.Any], columns: str) -> pd.DataFrame:
    """
    Calculate Variable Importance in Projection (VIP) scores for PLS.

    Parameters:
    - pls: Fitted PLSRegression model.
    - X: Input data used for PLS model training.

    Returns:
    - vip_scores: VIP scores for each feature.
    """
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    # Calculate sum of squared scores
    s = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    W_norm = np.sqrt(np.sum(w**2, axis=0))

    # Calculate VIP scores
    vip_scores = np.sqrt(X.shape[1] * (np.sum((w**2) * s, axis=1) / np.sum(s)))
    return pd.DataFrame({"Feature": columns, "VIP Score": vip_scores})

