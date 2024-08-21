import pandas as pd
import numpy as np
import processingfuncs as pf
import json
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import joblib
from datetime import datetime
from pathlib import Path
import time


def calcResults(model, model_type: str, X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot) -> pd.DataFrame:
    
    start = time.time()
    train_preds = model.predict(X_cold_train_clean).clip(0,None)
    test_preds = model.predict(X_cold_test_clean).clip(0,None)
    hot_preds = model.predict(X_hot_clean).clip(0,None)
    stop = time.time()

    results = [[
        model_type,
        mean_squared_log_error(y_cold_train, train_preds),
        mean_squared_log_error(y_cold_test, test_preds),
        mean_squared_log_error(y_hot, hot_preds),
        r2_score(y_cold_train, train_preds),
        r2_score(y_cold_test, test_preds),
        r2_score(y_hot, hot_preds),
        stop - start
    ]]

    output_df = pd.DataFrame(results, columns=['model', 'Training RMSE', 'Testing RMSE', 'Extrapolation RMSE', 'Training R2', 'Testing R2', 'Extrapolation R2', 'Score Time']).set_index('model')
    return output_df


def trainXGBoost(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str) -> XGBRegressor:
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = XGBRegressor(
        objective = 'reg:squarederror',
        tree_method = 'hist',
        device="cuda",
        verbosity=0
    )
    grid = {
        # 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'max_depth': [4, 5, 6, 7, 8],
        'min_child_weight': np.arange(0.0001, 0.5, 0.001),
        'gamma': np.arange(0.0,40.0,0.005),
        'learning_rate': np.arange(0.0005,0.3,0.0005),
        'subsample': np.arange(0.01,1.0,0.01),
        'colsample_bylevel': np.round(np.arange(0.1,1.0,0.01)),
        'colsample_bytree': np.arange(0.1,1.0,0.01),
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')
    booster = model.get_booster()

    with open(job_folder + 'xgb_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(booster, f)

    return model


def trainGBLinear(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str) -> XGBRegressor:
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = XGBRegressor(
        objective = 'reg:squarederror',
        booster='gblinear',
        device="cuda",
        verbosity=0
    )

    grid = {
        'lambda': np.logspace(-6, -1, 50), #L2 regularization term
        'alpha': np.logspace(-6, -1, 50), #L1 regularization term
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')
    booster = model.get_booster()

    with open(job_folder + 'gbl_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(booster, f)

    return model

def trainRandomForest(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = RandomForestRegressor(
        verbose=0,
        bootstrap=True,
        criterion="squared_error"
    )

    grid = {
        "n_estimators": np.arange(10, 100, 10),
        "max_depth": [3, 5, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": np.arange(1, 10, 2),
        "max_features": [0.5, 1, "sqrt", "log2"],
        "max_samples":np.arange(.1, .5, .1)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'rf_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainAdaBoostTree(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3)
    )

    grid = {
        "n_estimators": np.arange(30, 130, 10),
        'learning_rate': np.logspace(-4, 2, 50)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'adatree_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainAdaBoostOLS(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = AdaBoostRegressor(LinearRegression()
    )

    grid = {
        "n_estimators": np.arange(30, 130, 10),
        'learning_rate': np.logspace(-4, 2, 50)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'adaols_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainOrdinaryLeastSquares(X_train: pd.DataFrame, y_train: pd.DataFrame, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    model = LinearRegression(n_jobs=-1)
    
    model.fit(X_train, y_train)

    with open(job_folder + 'ols_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainRidge(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = Ridge(
    )

    grid = {
        'alpha': np.logspace(-4, 2, 50)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'ridge_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainLasso(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = Lasso(
    )

    grid = {
        'alpha': np.logspace(-4, 2, 50)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'lasso_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def trainElasticNet(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, job_folder: str):
    Path(job_folder).mkdir(parents=True, exist_ok=True)

    estimator = ElasticNet(
    )

    grid = {
        'alpha': np.logspace(-4, 2, 50),
        'l1_ratio': np.arange(.01, 1, .01)
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring = ('r2', 'neg_root_mean_squared_error'),
        refit='neg_root_mean_squared_error',
        n_jobs = -1,
        cv = 3,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    model = random_search.best_estimator_
    results_df.to_csv(job_folder + 'random_search_results.csv')

    with open(job_folder + 'elasticnet_model_cold_baseline.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model