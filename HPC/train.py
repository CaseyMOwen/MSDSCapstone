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
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime
from pathlib import Path


url = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/national/parquet/Baseline/baseline_metadata_and_annual_results.parquet'

df = pd.read_parquet(url)
# df = df.head(10)
# df.to_parquet('first_ten.parquet')

# df = pd.read_parquet('first_ten.parquet')
# pd.Series(df.columns).to_csv('all_columns.csv')
today = datetime.today().strftime('%Y-%m-%d')
Path(today).mkdir(parents=True, exist_ok=True)

in_cols = [col for col in df.columns if col.startswith('in')]
out_cols = [col for col in df.columns if col.startswith('out')]
df['out.electricity.total.energy_consumption.kwh'] = df['out.electricity.total.energy_consumption.kwh'].clip(0,None)
X, y = df[in_cols], df['out.electricity.total.energy_consumption.kwh']
X = pf.preprocess_columns(X)

## Build Category Encoders from those specified column_plan.csv
preprocessor = pf.build_column_transformer()

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', MinMaxScaler()),
    ('pca', PCA())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train_clean = pipeline.fit_transform(X_train, y_train)
X_test_clean = pipeline.transform(X_test)
X_clean = pipeline.transform(X)

with open(today + '/pipeline_pca.pkl', 'wb') as f:
    pickle.dump(pipeline, f)


xgb_estimator = XGBRegressor(
    objective = 'reg:squarederror',
    tree_method = 'hist',
    # enable_categorical = True
    device="cuda",
    verbosity=0
)

# parameters = {
#     'max_depth': list(range(1,5)),
#     'n_estimators': np.linspace(50, 250, 5, dtype=int),
#     'learning_rate': np.logspace(-2, -.3, 5)
# }


xgb_grid = {
        # 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'max_depth': [4, 5, 6],
        'min_child_weight': np.arange(0.0001, 0.5, 0.001),
        'gamma': np.arange(0.0,40.0,0.005),
        'learning_rate': np.arange(0.0005,0.3,0.0005),
        'subsample': np.arange(0.01,1.0,0.01),
        'colsample_bylevel': np.round(np.arange(0.1,1.0,0.01)),
        'colsample_bytree': np.arange(0.1,1.0,0.01),
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_estimator,
    param_distributions=xgb_grid,
    n_iter=200,
    scoring = ('r2', 'neg_root_mean_squared_error'),
    refit='neg_root_mean_squared_error',
    n_jobs = -1,
    cv = 3,
    verbose=True,
    pre_dispatch='1*n_jobs',
    return_train_score=True
)

# pf.random_search(parameters, X_train_clean, y_train, cv=3, n_iter=100, results_path='random_search_results.csv')


# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=parameters,
#     scoring = 'neg_root_mean_squared_error',
#     n_jobs = -1,
#     cv = 3,
#     verbose=True,
#     pre_dispatch='1*n_jobs',
#     return_train_score=True
# )

xgb_random_search.fit(X_train_clean, y_train)
xgb_results_df = pd.DataFrame(xgb_random_search.cv_results_)
xgb_model = xgb_random_search.best_estimator_
xgb_results_df.to_csv(today + '/xgb_random_search_results.csv')


# model = XGBRegressor(
#     objective = 'reg:squarederror',
#     tree_method = 'hist',
#     max_depth = 2,
#     n_estimators = 50,
#     learning_rate = .1,
#     device="cuda"
# )

# model.fit(X_train_clean, y_train)

preds = xgb_model.predict(X_test_clean)
preds = preds.clip(0,None)
rmse = mean_squared_log_error(y_test, preds)

print(f"RMSE of the best xgb model: {rmse:.5f}")

# Make the final model on the full dataset

xgb_model.fit(X_clean,y)
booster = xgb_model.get_booster()
booster.save_model(today + '/xgb_pca_model_baseline.json')
'''
##Repeat for Random Forest

rf_estimator = RandomForestRegressor(
    # enable_categorical = True
    verbose=0,
    bootstrap=True
)

rf_grid = {
    "n_estimators": np.arange(10, 200, 10),
    "max_depth": [3, 5, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": np.arange(1, 10, 2),
    "max_features": [0.5, 1, "sqrt", "log2"],
    "max_samples":[0.25, 0.5, 0.75]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf_estimator,
    param_distributions=rf_grid,
    n_iter=50,
    scoring = ('r2', 'neg_root_mean_squared_error'),
    refit='neg_root_mean_squared_error',
    n_jobs = -1,
    cv = 3,
    verbose=True,
    pre_dispatch='1*n_jobs',
    return_train_score=True
)

# pf.random_search(parameters, X_train_clean, y_train, cv=3, n_iter=100, results_path=today + '/random_search_results.csv')


# grid_search = GridSearchCV(
#     estimator=estimator,
#     param_grid=parameters,
#     scoring = 'neg_root_mean_squared_error',
#     n_jobs = -1,
#     cv = 3,
#     verbose=True,
#     pre_dispatch='1*n_jobs',
#     return_train_score=True
# )

rf_random_search.fit(X_train_clean, y_train)
rf_results_df = pd.DataFrame(rf_random_search.cv_results_)
rf_model = rf_random_search.best_estimator_
rf_results_df.to_csv(today + '/pca_rf_random_search_results.csv')


# model = XGBRegressor(
#     objective = 'reg:squarederror',
#     tree_method = 'hist',
#     max_depth = 2,
#     n_estimators = 50,
#     learning_rate = .1,
#     device="cuda"
# )

# model.fit(X_train_clean, y_train)

preds = rf_model.predict(X_test_clean)
preds = preds.clip(0,None)
rmse = mean_squared_log_error(y_test, preds)

print(f"RMSE of the best rf model: {rmse:.5f}")

# Make the final model on the full dataset
rf_model.fit(X_clean,y)
joblib.dump(rf_model,"today + '/rf_model_baseline.joblib")
'''