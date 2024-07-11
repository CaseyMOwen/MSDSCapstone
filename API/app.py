from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import xgboost as xgb
import os
import pickle
import numpy as np
import processingfuncs as pf

app = Flask(__name__)
CORS(app)

def get_avg_home():
    with open('appfiles/avg_home.json') as f: 
        data = f.read() 
    
    combined_dicts = json.loads(data)
    avg_home_df = pd.DataFrame(combined_dicts['means'], index=[0])
    for column in avg_home_df:
        avg_home_df[column] = avg_home_df[column].astype(combined_dicts['types'][column])
    return avg_home_df

def sample_home(feats: dict, n_homes:int):
    state = feats["in.state"]
    url = f'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/by_state/state={state}/parquet/Baseline/{state}_baseline_metadata_and_annual_results.parquet'
    df = pd.read_parquet(url)
    in_cols = [col for col in df.columns if col.startswith('in')]
    X = df[in_cols]
    X = pf.preprocess_columns(X)
    # query_str = ' & '.join([repr(f'`{key}` == "{value}"') for key, value in feats.items()])
    # query_str = '`in.geometry_floor_area` == "3000-3999"'
    # print(query_str)
    # filtered_df = df.query(query_str)
    X_filtered = X.loc[(X[list(feats)] == pd.Series(feats)).all(axis=1)]
    X_sampled = X_filtered.sample(n=n_homes, random_state=42)
    return X_sampled



def set_feature(var, val, input_df: pd.DataFrame):
#     var = 'in.ashrae_iecc_climate_zone_2004'
# val = '2A'

    old_type = str(input_df[var].dtype)
    if old_type == 'category' and val not in input_df[var].cat.categories:
        input_df[var] = input_df[var].cat.add_categories([val])

    # avg_home_df['in.ashrae_iecc_climate_zone_2004'].cat.add_categories('2A')
    input_df.at[0, var] = val
    input_df[var] = input_df[var].astype(old_type)
    return input_df

def predict(input_df):
    with open('appfiles/pipeline_pca.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    X = pipeline.transform(input_df)
    booster = xgb.Booster()
    # model = xgb.XGBRegressor()
    booster.load_model('appfiles/models/xgb_pca_model_baseline.json')
    dmatrix = xgb.DMatrix(X)
    return booster.predict(dmatrix)



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # feats_df = get_avg_home()
    if request.data:
        feats = request.json  # Expecting JSON input
    else:
        feats = {}
    print(feats)
    samples_df = sample_home(feats, 10)
    predictions = predict(samples_df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
