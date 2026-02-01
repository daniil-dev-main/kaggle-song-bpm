import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def skl_poly_lr(train_x: pd.DataFrame, train_y: pd.Series, val_x: pd.DataFrame) -> tuple[LinearRegression, np.ndarray]:
    """
        The function creates second-degree polynomial features from the
        training data, fits a linear regression model, and applies the
        same transformation to the validation data.

        Args:
            train_x (pd.DataFrame): Training feature matrix of shape (n_samples, n_features).
            train_y (pd.Series): Target values for training of shape (n_samples, 1).
            val_x (pd.DataFrame): Validation feature matrix to be transformed.

        Returns:
            tuple:
                - model (LinearRegression): Fitted linear regression model.
                - val_x_poly (np.ndarray): Polynomial-transformed validation features.
        """
    poly = PolynomialFeatures(degree=2)
    x_poly_t = poly.fit_transform(train_x)
    model = LinearRegression()
    model.fit(x_poly_t, train_y)
    return model, poly.fit_transform(val_x)

def lin_reg(train_x: pd.DataFrame, train_y: pd.Series) -> np.ndarray:
    """
        This function estimates regression coefficients using the Moore–Penrose pseudoinverse.

        Args:
            train_x (pd.DataFrame): Training feature matrix of shape (n_samples, n_features).
            train_y (pd.Series): Target values of shape (n_samples, 1).

        Returns:
            np.ndarray: Estimated weight vector of shape (n_features, 1).
        """
    x = train_x.values
    y = train_y.values
    w = np.linalg.pinv(x.T @ x) @ x.T @ y
    print(pd.DataFrame({ "Feature": train_x.columns, "Weight": w}))
    return w

def xgb_random_search(train_x: pd.DataFrame, train_y: pd.Series, n_iter: int = 20, random_state: int = 42) -> XGBRegressor:
    """
        Performs randomized hyperparameter search for an XGBoost regressor.

        The function uses RandomizedSearchCV to search over a predefined
        hyperparameter space and returns the best-performing model based
        on cross-validated mean squared error.

        Args:
            train_x (pd.DataFrame): Training feature matrix of shape (n_samples, n_features).
            train_y (pd.Series): Target values of shape (n_samples, 1).
            n_iter (int, optional): Number of parameter settings sampled.
                Defaults to 20.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.

        Returns:
            XGBRegressor: Best estimator found during randomized search.
        """
    param_distributions = {
        "n_estimators": [100, 300, 500, 700],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": uniform(0.01, 0.09),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 0.5),
        "reg_lambda": uniform(1, 1),
    }

    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=random_state,
    )

    search.fit(train_x, train_y)
    print("Best hyperparameters:", search.best_params_)

    return search.best_estimator_