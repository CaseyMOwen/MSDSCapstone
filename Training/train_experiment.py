import pandas as pd
# import numpy as np
# import processingfuncs as pf
# import json
# import category_encoders as ce
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_log_error, r2_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import KFold
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor
# import joblib
from datetime import datetime
from pathlib import Path
import modelfunctions as mf
import os
import pipelineclasses as pc



# output = 'out.electricity.total.energy_consumption.kwh'
output = 'out.site_energy.total.energy_consumption.kwh'

# url = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/national/parquet/Baseline/baseline_metadata_and_annual_results.parquet'
url = 'first_500.parquet'

column_plan_df = pd.read_csv('column_plan.csv',usecols=['field_name', 'keep_for_model'])
columns = column_plan_df.loc[
    (column_plan_df['keep_for_model'] == 'Yes') | 
    (column_plan_df['keep_for_model'] == 'Split')
]['field_name'].to_list()
columns.append(output)

df = pd.read_parquet(url, columns=columns, engine='pyarrow')
today = 'Jobs/' + datetime.today().strftime('%Y-%m-%d') + '_train'
Path(today).mkdir(parents=True, exist_ok=True)

# in_cols = [col for col in df.columns if col.startswith('in')]
# df[output] = df[output].clip(0,None)

hottest_counties = pd.read_csv('TMY3_aggregates.csv').sort_values('avg temp', ascending=False).head(500)['gisjoin'].to_list()

df_hot = df[df['in.county'].isin(hottest_counties)]
df_cold = df[~df['in.county'].isin(hottest_counties)]

X_hot, y_hot = df_hot.drop(columns=[output]), df_hot[output].clip(0,None)
# X_hot = pf.preprocess_columns(X_hot)
# X_hot = pf.add_tmy3_data(X_hot)

X_cold, y_cold = df_cold.drop(columns=[output]), df_cold[output].clip(0,None)
# X_cold = pf.preprocess_columns(X_cold)
# X_cold = pf.add_tmy3_data(X_cold)


## Build Category Encoders from those specified column_plan.csv
# preprocessor = pf.build_column_transformer()

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', pc.Preprocessing()),
    ('addweather', pc.AddWeatherData(year_range='current')),
    ('encoder', pc.Encoding()),
    ('scaler', MinMaxScaler())
    # ('pca', PCA())
])


X_cold_train, X_cold_test, y_cold_train, y_cold_test = train_test_split(X_cold, y_cold, random_state=1)

X_cold_train_clean = pipeline.fit_transform(X_cold_train, y_cold_train)
X_cold_test_clean = pipeline.transform(X_cold_test)
X_hot_clean = pipeline.transform(X_hot)
# X_clean = pipeline.transform(X)

nan_count_cold = X_cold.isna().sum()
print("NaN count on X_cold:")
print(nan_count_cold[nan_count_cold > 0])

nan_count_hot = X_hot.isna().sum()
print("NaN count on X_hot:")
print(nan_count_hot[nan_count_hot > 0])

with open(today + '/pipeline_cold.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Run extrapolation study on selected models
output_file = today + '/full_results.csv'


## XGBoost
xgb_model = mf.trainXGBoost(X_cold_train_clean, y_cold_train, 50, today + '/xgboost/')
results_df = mf.calcResults(xgb_model, "xgboost", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

# Error apparently giving out NaN results on non-NaN inputs when evaluating
## Gblinear
# gbl_model = mf.trainGBLinear(X_cold_train_clean, y_cold_train, 50, today + '/gblinear/')
# results_df = mf.calcResults(gbl_model, "gblinear", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
# results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## AdaBoost Tree
adatree_model = mf.trainAdaBoostTree(X_cold_train_clean, y_cold_train, 50, today + '/adaboosttree/')
results_df = mf.calcResults(adatree_model, "adaboosttree", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## AdaBoost OLS
adaols_model = mf.trainAdaBoostOLS(X_cold_train_clean, y_cold_train, 50, today + '/adaboostols/')
results_df = mf.calcResults(adaols_model, "adaboostols", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))


## Ordinary Least Squares Linear Regression
ols_model = mf.trainOrdinaryLeastSquares(X_cold_train_clean, y_cold_train, today + '/ordinaryleastsquares/')
results_df = mf.calcResults(ols_model, "ordinaryleastsquares", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Ridge Regression
ridge_model = mf.trainRidge(X_cold_train_clean, y_cold_train, 50, today + '/ridge/')
results_df = mf.calcResults(ridge_model, "ridge", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Lasso Regression
lasso_model = mf.trainLasso(X_cold_train_clean, y_cold_train, 50, today + '/lasso/')
results_df = mf.calcResults(lasso_model, "lasso", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Elastic Net Regression
en_model = mf.trainElasticNet(X_cold_train_clean, y_cold_train, 50, today + '/elasticnet/')
results_df = mf.calcResults(en_model, "elasticnet", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

## Random Forest
rf_model = mf.trainRandomForest(X_cold_train_clean, y_cold_train, 50, today + '/randomforest/')
results_df = mf.calcResults(rf_model, "randomforest", X_cold_train_clean, X_cold_test_clean, X_hot_clean, y_cold_train, y_cold_test, y_hot)
results_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

# Also try SGDRegressor

# Make the final model on the full dataset

# xgb_model.fit(X_clean,y)