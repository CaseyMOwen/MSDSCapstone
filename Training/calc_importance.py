import pickle
import pandas as pd
import processingfuncs as pf
from sklearn.inspection import permutation_importance
import xgboost as xgb
from sklearn.pipeline import Pipeline
from datetime import datetime
from pathlib import Path


url = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/national/parquet/Baseline/baseline_metadata_and_annual_results.parquet'

today = 'Jobs/' + datetime.today().strftime('%Y-%m-%d') + '_importance'
Path(today).mkdir(parents=True, exist_ok=True)

# output = 'out.electricity.total.energy_consumption.kwh'
output = 'out.site_energy.total.energy_consumption.kwh'


df = pd.read_parquet(url)
# df = pd.read_parquet('first_500.parquet')
in_cols = [col for col in df.columns if col.startswith('in')]
out_cols = [col for col in df.columns if col.startswith('out')]
df[output] = df[output].clip(0,None)

X, y = df[in_cols], df[output]
X = pf.preprocess_columns(X)
X = pf.add_tmy3_data(X)


with open('Jobs/2024-08-10_train/xgboost/xgb_model_cold_baseline.pkl', 'rb') as f:
        model = pickle.load(f)

with open('Jobs/2024-08-10_train/pipeline_cold.pkl', 'rb') as f:
        pipeline = pickle.load(f)

class convert_to_dmatrix():
    def __init__(self) -> None:
           pass
    
    def transform(self, X):
           return xgb.DMatrix(X)

converter = convert_to_dmatrix()

pipeline.steps.append(['converter', converter])
pipeline.steps.append(['estimator', model])
# pipeline.named_steps
result = permutation_importance(pipeline, X, y, n_jobs=10, random_state=1, scoring='neg_root_mean_squared_error')

with open(today + '/importance_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# X_clean = pipeline.transform(X)


# feature_important = model.get_score(importance_type='weight')
# keys = list(feature_important.keys())
# values = list(feature_important.values())

# data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
# data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features