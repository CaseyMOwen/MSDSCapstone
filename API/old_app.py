from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import xgboost as xgb
import os
import pickle
import numpy as np
import processingfuncs as pf
import time
from urllib.parse import quote_plus
from sqlalchemy.engine import create_engine
import pandas as pd
from config import Aws as aws
from pyathena import connect
from pyathena.pandas.cursor import PandasCursor

app = Flask(__name__)
CORS(app)

def create_athena_connection():
    # query = "SELECT * FROM \"resstock\".\"resstock_dataset_2024.1parquet\" WHERE upgrade='0' limit 9;"

    conn_str = (
        "awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@"
        "athena.{region_name}.amazonaws.com:443/"
        "{schema_name}?s3_staging_dir={s3_staging_dir}&work_group=primary"
    )

    # Create SQLAlchemy connection with pyathena
    engine = create_engine(
        conn_str.format(
            aws_access_key_id=quote_plus(aws.ACCESS_KEY),
            aws_secret_access_key=quote_plus(aws.SECRET_KEY),
            region_name=aws.REGION,
            schema_name=aws.SCHEMA_NAME,
            s3_staging_dir=quote_plus(aws.S3_STAGING_DIR),
        )
    )

    athena_connection = engine.connect()
    return athena_connection


def query_athena(query):
    # query = 'SELECT * FROM "resstock"."release_2024.1by_state" WHERE "state"=\'FL\' AND "measure_group"=\'Baseline\' AND "in.county"=\'G1200090\' limit 10'
    cursor = connect(aws_access_key_id=aws.ACCESS_KEY,
                    aws_secret_access_key=aws.SECRET_KEY,
                    s3_staging_dir=aws.S3_STAGING_DIR,
                    region_name=aws.REGION,
                    cursor_class=PandasCursor).cursor()
    df = cursor.execute(query).as_pandas()
    print(f"Queue time: {cursor.query_queue_time_in_millis/1000}")
    print(f"Execution time: {cursor.total_execution_time_in_millis/1000}")

    return df

def get_avg_home():
    with open('appfiles/avg_home.json') as f: 
        data = f.read() 
    
    combined_dicts = json.loads(data)
    avg_home_df = pd.DataFrame(combined_dicts['means'], index=[0])
    for column in avg_home_df:
        avg_home_df[column] = avg_home_df[column].astype(combined_dicts['types'][column])
    return avg_home_df

def sample_home(feats: dict, n_homes:int):
    start = time.time()
    state = feats["in.state"]
    gisjoin = feats["in.county"]
    year_range = feats["year_range"]
    del feats["year_range"]
    del feats["in.county"]
    step1 = time.time()
    url = f'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/by_state/state={state}/parquet/Baseline/{state}_baseline_metadata_and_annual_results.parquet'
    df = pd.read_parquet(url)
    # conn = create_athena_connection()
    measure_group = "Baseline"
    query = f'SELECT * FROM "resstock"."release_2024.1by_state" WHERE "state"=\'{state}\' AND "measure_group"=\'{measure_group}\' AND "in.county"=\'{gisjoin}\' limit {n_homes}'
    # df = pd.read_sql_query(query, conn)
    # df = query_athena(query)
    step1_5 = time.time()
    in_cols = [col for col in df.columns if col.startswith('in')]
    X = df[in_cols]
    step2 = time.time()
    X = X[X["in.county"] == gisjoin]
    X = pf.preprocess_columns(X)
    step3 = time.time()
    X = pf.add_ftmy3_data(X, state, gisjoin, year_range)
    step4 = time.time()
    # query_str = ' & '.join([repr(f'`{key}` == "{value}"') for key, value in feats.items()])
    # query_str = '`in.geometry_floor_area` == "3000-3999"'
    # print(query_str)
    # filtered_df = df.query(query_str)
    X_filtered = X.loc[(X[list(feats)] == pd.Series(feats)).all(axis=1)]
    num_to_sample = min(n_homes, X_filtered.shape[0])
    print(f'Homes Sampled: {num_to_sample}')
    X_sampled = X_filtered.sample(n=num_to_sample, random_state=42)
    # print(X_sampled[['in.county', 'avg temp']])
    stop = time.time()
    print(f'\tDatabase Connection: {step1 - start}')
    print(f'\tQuery Database: {step1_5 - step1}')
    print(f'\tSelect in cols: {step2 - step1_5}')
    print(f'\tPreprocess Columns: {step3 - step2}')
    print(f'\tAdd fTYM3 data: {step4 - step3}')    
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
    with open('appfiles/pipeline_cold.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    X = pipeline.transform(input_df)
    # booster = xgb.Booster()
    with open('appfiles/models/xgb_model_cold_baseline.pkl', 'rb') as f:
        booster = pickle.load(f)
    # booster.load_model('appfiles/models/xgb_pca_model_baseline.json')
    dmatrix = xgb.DMatrix(X)
    return booster.predict(dmatrix)



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    start = time.time()
    # feats_df = get_avg_home()
    if request.data:
        feats = request.json  # Expecting JSON input
    else:
        feats = {}
    # print(feats)
    step1 = time.time()
    print(f"Load Inputs: {step1 - start}")
    samples_df = sample_home(feats, 100)
    step2 = time.time()
    print(f"Sample Home: {step2 - step1}")
    predictions = predict(samples_df)
    step3 = time.time()
    print(f"Predict: {step3 - step2}")
    result = jsonify(predictions.tolist())
    stop = time.time()
    print(f"Turn Predictions to list: {stop - step3}")
    print(f"Total Time: {stop - start}")
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
