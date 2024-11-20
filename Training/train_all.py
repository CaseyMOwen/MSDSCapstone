import pandas as pd
import numpy as np
# import processingfuncs as pf
# import json
# import category_encoders as ce
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import KFold
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor
# import joblib
from datetime import datetime
from pathlib import Path
# import modelfunctions as mf
import os
import pipelineclasses as pc
import time

'''
# Baseline
output = 'out.electricity.total.energy_consumption.kwh'
output = 'out.site_energy.total.energy_consumption.kwh'
# For measures
output = 'out.electricity.total.energy_consumption.kwh.savings'
output = 'out.site_energy.total.energy_consumption.kwh.savings'
'''
# outputs_dict = {
#     "Baseline": {
#         'electricity':'out.electricity.total.energy_consumption.kwh', 'total':'out.site_energy.total.energy_consumption.kwh'},
#     "Measure": {
#         'electricity':'out.electricity.total.energy_consumption.kwh.savings', 'total':'out.site_energy.total.energy_consumption.kwh.savings'} 
# }

measures_df = pd.read_csv('measures.csv')
# column_plan_df = pd.read_csv('column_plan.csv',usecols=['field_name', 'keep_for_model'])
# in_cols = column_plan_df.loc[
#     (column_plan_df['keep_for_model'] == 'Yes') | 
#     (column_plan_df['keep_for_model'] == 'Split')
# ]['field_name'].to_list()

job_path = 'Jobs/' + datetime.today().strftime('%Y-%m-%d') + '_train_all'
Path(job_path).mkdir(parents=True, exist_ok=True)

def trainModel(df, output_col:str, job_folder:str, n_iter:int, measure_code:str, version:str):
    # column_plan_df = pd.read_csv('column_plan.csv',usecols=['field_name', 'keep_for_model'])
    # columns = in_cols
    # columns.append(output_col)
    
    # All irrelevant columns are dropped as part of pipeline
    y = df[output_col]

    # Build pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', pc.Preprocessing(version=version)),
        ('addweather', pc.AddWeatherData(year_range='current')),
        ('encoder', pc.Encoding(version=version)),
        ('scaler', MinMaxScaler())
        # ('pca', PCA())
    ])

    X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1)
    X_train_clean = pipeline.fit_transform(X_train, y_train)
    X_test_clean = pipeline.transform(X_test)
    # X_clean = pipeline.transform(X)
    
    # nan_count = X.isna().sum()
    # print("NaN count on X:")
    # print(nan_count[nan_count > 0])

    with open(job_folder + '/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    

    estimator = XGBRegressor(
        objective = 'reg:squarederror',
        tree_method = 'hist',
        device="cuda",
        verbosity=0
    )
    grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # 'max_depth': [4, 5, 6, 7, 8],
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
        cv = 5,
        verbose=True,
        pre_dispatch='1*n_jobs',
        return_train_score=True
    )
    
    random_search.fit(X_train_clean, y_train)
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv(job_folder + '/random_search_results.csv')
    model = random_search.best_estimator_

    train_preds = model.predict(X_train_clean)
    test_preds = model.predict(X_test_clean)
    
    results = [[
        measure_code,
        output_type,
        mean_squared_error(y_train, train_preds),
        mean_squared_error(y_test, test_preds),
        r2_score(y_train, train_preds),
        r2_score(y_test, test_preds),
    ]]

    scores_df = pd.DataFrame(results, columns=['Measure', 'Output Type', 'Training RMSE', 'Testing RMSE', 'Training R2', 'Testing R2']).set_index('Measure')
    
    booster = model.get_booster()
    scores_df.to_csv(job_folder + '/scores.csv')

    with open(job_folder + '/xgb_model.pkl', 'wb') as f:
        pickle.dump(booster, f)

for i, row in measures_df.iterrows():
    if row['name'] == "Baseline":
        # measure_type = 'Baseline'
        # The column names are slightly different depending on if it is the baseline or a measure
        elec_col = 'out.electricity.total.energy_consumption.kwh'
        total_col = 'out.site_energy.total.energy_consumption.kwh'
        other_fuel_col = 'out.other_fuel.total.energy_consumption.kwh'
    else:
        # measure_type = "Measure"
        elec_col = 'out.electricity.total.energy_consumption.kwh.savings'
        total_col = 'out.site_energy.total.energy_consumption.kwh.savings'
        other_fuel_col = 'out.other_fuel.total.energy_consumption.kwh.savings'
    print('Loading Dataset...')
    start = time.time()
    df = pd.read_parquet(row['parquet_url'], engine='pyarrow')
    # df = pd.read_csv(row['csv_url'], nrows=100)
    stop = time.time()
    print(f'Dataset Loaded in {stop - start}s')
    # df = df.head(100)

    # Create new column representing all non-electricity use in kwh
    df[other_fuel_col] = df[total_col] - df[elec_col]
    output_cols = {"electricity": elec_col, "other_fuel": other_fuel_col}
    for output_type in output_cols:
        measure_code = row['folder_name']
        print(f'Training model {measure_code} with output type {output_type}')
        measure_folder = os.path.join(job_path, measure_code)
        Path(measure_folder).mkdir(parents=True, exist_ok=True)
        job_folder = os.path.join(measure_folder, output_type)
        Path(job_folder).mkdir(parents=True, exist_ok=True)
        trainModel(df, output_cols[output_type], job_folder, 200, measure_code, row["resstock_version"])

'''
# Run extrapolation study on selected models
output_file = job_path + '/full_results.csv'


## XGBoost
xgb_model = mf.trainXGBoost(X_cold_train_clean, y_cold_train, 50, job_path + '/xgboost/')
results_df = mf.calcResults(xgb_model, "xgboost", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

# Error apparently giving out NaN results on non-NaN inputs when evaluating
## Gblinear
# gbl_model = mf.trainGBLinear(X_cold_train_clean, y_cold_train, 50, job_path + '/gblinear/')
# results_df = mf.calcResults(gbl_model, "gblinear", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
# results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## AdaBoost Tree
adatree_model = mf.trainAdaBoostTree(X_cold_train_clean, y_cold_train, 50, job_path + '/adaboosttree/')
results_df = mf.calcResults(adatree_model, "adaboosttree", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## AdaBoost OLS
adaols_model = mf.trainAdaBoostOLS(X_cold_train_clean, y_cold_train, 50, job_path + '/adaboostols/')
results_df = mf.calcResults(adaols_model, "adaboostols", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))


## Ordinary Least Squares Linear Regression
ols_model = mf.trainOrdinaryLeastSquares(X_cold_train_clean, y_cold_train, job_path + '/ordinaryleastsquares/')
results_df = mf.calcResults(ols_model, "ordinaryleastsquares", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Ridge Regression
ridge_model = mf.trainRidge(X_cold_train_clean, y_cold_train, 50, job_path + '/ridge/')
results_df = mf.calcResults(ridge_model, "ridge", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Lasso Regression
lasso_model = mf.trainLasso(X_cold_train_clean, y_cold_train, 50, job_path + '/lasso/')
results_df = mf.calcResults(lasso_model, "lasso", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Elastic Net Regression
en_model = mf.trainElasticNet(X_cold_train_clean, y_cold_train, 50, job_path + '/elasticnet/')
results_df = mf.calcResults(en_model, "elasticnet", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Random Forest
rf_model = mf.trainRandomForest(X_cold_train_clean, y_cold_train, 50, job_path + '/randomforest/')
results_df = mf.calcResults(rf_model, "randomforest", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

# Also try SGDRegressor

# Make the final model on the full dataset

# xgb_model.fit(X_clean,y)
'''