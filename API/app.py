from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import xgboost as xgb
import os
import pickle
import time
from sklearn.pipeline import Pipeline
# from urllib.parse import quote_plus
# from sqlalchemy.engine import create_engine
import pandas as pd
# from config import Aws as aws
# from pyathena import connect
# from pyathena.pandas.cursor import PandasCursor
import pipelineclasses as pc
import numpy as np
import polars as pl
import re
import copy

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if request.data:
        feats = request.json  # Expecting JSON input
    else:
        feats = {}
    start = time.time()
    # X = sample_home(feats, 100)
    # state = feats["in.state"]
    # gisjoin = feats["in.county_and_puma"].split(", ")[0]
    year_range = feats.pop("year_range")
    if "num_samples" in feats:
        num_samples = feats.pop("num_samples")
    else:
        num_samples = 100
    feats = lookup_county_puma(feats)
    feats_2022_1 = copy.deepcopy(feats)
    # 2022 does not have support for distinguising non-ducted geat pumps from ducted heat pumps - merge both types into one "heat pump" feature
    if "in.hvac_heating_type" in feats_2022_1 and feats_2022_1["in.hvac_heating_type"] == "Non-Ducted Heat Pump":
        feats_2022_1["in.hvac_heating_type"] = 'Ducted Heat Pump'
    print(feats)
    print(feats_2022_1)
    X_2022, state, gisjoin = generate_sample(feats_2022_1, num_samples, release="2022_1")
    step1 = time.time()
    X_2024, state, gisjoin = generate_sample(feats, num_samples, release="2024_1")
    step2 = time.time()
    X_2022_df, X_2024_df = X_2022.to_pandas(), X_2024.to_pandas()
    print(X_2024.select('in.duct_leakage_and_insulation').head(5))
    print(X_2024_df['in.duct_leakage_and_insulation'].head(5))
    

    output = {'cost': {'electricity': .15, 'other_fuel': 1}, 'baseline': {'1980-1999': {}, '2000-2019': {}, '2020-2039': {}, '2040-2059': {}, '2060-2079': {}, '2080-2099': {}}, 'measures': {}}
    # TODO: possible read files in parallel, then make predictions on loaded objects
    THERM_FACTOR = 0.0341214116
    measures_df = pd.read_csv('measures.csv')
    for measure_folder in os.scandir('appfiles/models'):
        # print(subfolder)
        measure_name = measure_folder.name
        is_baseline = (measure_name == "2024_1_0")
        measure_row = measures_df[measures_df['folder_name'] == measure_name].to_dict('records')[0]
        id = measure_row['measure_id']
        if not is_baseline:
            output['measures'][id] = {}
            output['measures'][id]['name'] = measure_row['name']
            output['measures'][id]['code'] = measure_name
            output['measures'][id]['description'] = measure_row['description']
        for model_folder in os.scandir(measure_folder):
            model_type = model_folder.name
            print(measure_name + '_' + model_type)
            with open(model_folder.path + '/pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            with open(model_folder.path + '/xgb_model.pkl', 'rb') as f:
                booster = pickle.load(f)
            if measure_name.startswith('2024_1'):
                X = X_2024_df
            elif measure_name.startswith('2022_1'):
                X = X_2022_df
            if is_baseline:
                # year_ranges = ['1980-1999', '2000-2019', '2020-2039', '2040-2059', '2060-2079', '2080-2099'] 
                for yr in output['baseline']:
                    # output['baseline'][yr] = {}
                    predictions = get_predictions(X, booster, pipeline, yr, state, gisjoin)
                    if model_type == 'other_fuel':
                        # Convert kWh to therms
                        predictions = predictions * THERM_FACTOR
                    output['baseline'][yr][model_type] = predictions.tolist()
            else:
                predictions = get_predictions(X, booster, pipeline, year_range, state, gisjoin)
                if model_type == 'other_fuel':
                    # Convert kWh to therms
                    predictions = predictions * THERM_FACTOR
                output['measures'][id][model_type] = predictions.tolist()
    stop = time.time()
    # TODO: only consider homes where the measure is applicable
    # for model in preds_dict:
        # Subtract electricity off of total and turn into fuel
        # Better - train new model on the difference of the columns, and call it "other fuel"
        # pass
    print(f'Generate 2022 Samples: {step1 - start}')
    print(f'Generate 2024 Samples: {step2 - step1}')
    print(f'Make predictions: {stop - step2}')
    print(f'Total: {stop - start}')


    return jsonify(output)

'''
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
    column_plan_df = pd.read_csv('column_plan.csv',usecols=['field_name', 'keep_for_model'])
    columns = column_plan_df.loc[
        (column_plan_df['keep_for_model'] == 'Yes') | 
        (column_plan_df['keep_for_model'] == 'Split')
    ]['field_name'].to_list()
    state = feats["in.state"]
    gisjoin = feats["in.county"]
    year_range = feats["year_range"]
    del feats["year_range"]
    # del feats["in.county"]
    url = f'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_dataset_2024.1/resstock_tmy3/metadata_and_annual_results/by_state/state={state}/parquet/Baseline/{state}_baseline_metadata_and_annual_results.parquet'
    filters = []
    for key in feats:
        filters.append((key, '==', feats[key]))
    X = pd.read_parquet(url, filters=filters, columns=columns, engine='pyarrow')
    step1 = time.time()
    print(f'Total rows loaded: {X.shape[0]}')
    # conn = create_athena_connection()
    measure_group = "Baseline"
    # query = f'SELECT * FROM "resstock"."release_2024.1by_state" WHERE "state"=\'{state}\' AND "measure_group"=\'{measure_group}\' AND "in.county"=\'{gisjoin}\' limit {n_homes}'
    # df = pd.read_sql_query(query, conn)
    # df = query_athena(query)
    step2 = time.time()
    # query_str = ' & '.join([repr(f'`{key}` == "{value}"') for key, value in feats.items()])
    # query_str = '`in.geometry_floor_area` == "3000-3999"'
    
    # May want to filter after - to handle too-specific features
    X_filtered = X.loc[(X[list(feats)] == pd.Series(feats)).all(axis=1)]
    num_to_sample = min(n_homes, X_filtered.shape[0])
    print(f'Homes Sampled: {num_to_sample}')
    X_sampled = X_filtered.sample(n=num_to_sample, random_state=42)
    step3 = time.time()

    with open('appfiles/pipeline_cold.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    pipeline.set_params(addweather__year_range=year_range, addweather__state=state, addweather__gisjoin=gisjoin)
    step4 = time.time()
    X_transformed = pipeline.transform(X_sampled)
    stop = time.time()
    print(f'\tLoad File: {step1 - start}')
    print(f'\tSelect in cols: {step2 - step1}')
    print(f'\tFilter and Sample: {step3 - step2}')
    print(f'\tLoad Pipeline: {step4 - step3}')
    print(f'\tPipeline Transform: {stop - step4}')
    return X_transformed
'''

def get_predictions(X_sampled: pd.DataFrame, booster:xgb.Booster, pipeline: Pipeline, year_range:str, state:str, gisjoin:str):
    # with open('appfiles/pipeline_cold.pkl', 'rb') as f:
    #     pipeline = pickle.load(f)
    pipeline.set_params(addweather__year_range=year_range, addweather__state=state, addweather__gisjoin=gisjoin)
    X_transformed = pipeline.transform(X_sampled)
    # with open('appfiles/models/xgb_model_cold_baseline.pkl', 'rb') as f:
    #     booster = pickle.load(f)
    # booster.load_model('appfiles/models/xgb_pca_model_baseline.json')
    dmatrix = xgb.DMatrix(X_transformed)
    return booster.predict(dmatrix)

'''
def transform_sample(X_sampled, year_range:str, state:str, gisjoin:str):
    with open('appfiles/pipeline_cold.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    pipeline.set_params(addweather__year_range=year_range, addweather__state=state, addweather__gisjoin=gisjoin)
    # step4 = time.time()
    X_transformed = pipeline.transform(X_sampled)
    # stop = time.time()
    # print(f'\tLoad File: {step1 - start}')
    # print(f'\tSelect in cols: {step2 - step1}')
    # print(f'\tFilter and Sample: {step3 - step2}')
    # print(f'\tLoad Pipeline: {step4 - step3}')
    # print(f'\tPipeline Transform: {stop - step4}')
    return X_transformed

def predict(X:pl.DataFrame):
    # with open('appfiles/pipeline_cold.pkl', 'rb') as f:
    #     pipeline = pickle.load(f)
    # X = pipeline.transform(input_df)
    # booster = xgb.Booster()
    with open('appfiles/models/xgb_model_cold_baseline.pkl', 'rb') as f:
        booster = pickle.load(f)
    # booster.load_model('appfiles/models/xgb_pca_model_baseline.json')
    dmatrix = xgb.DMatrix(X)
    return booster.predict(dmatrix)
'''
def to_underscore_case(s):
    # Replace '::' with '/'
    s = s.replace('::', '/')
    
    # Convert CamelCase to snake_case
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    
    # Replace '-' and ' ' with '_'
    s = s.replace('-', '_').replace(' ', '_')
    
    # Convert to lowercase
    s = s.lower()
    
    return "in." + s

# Feats is dictionary with unserscore case field names, and value as value

def lookup_county_puma(feats:dict):
    '''
    Replaces geoid with county and puma in feats dict
    '''
    if not "geoid" in feats:
        raise ValueError("feats must include at least geoid as a key")
    
    with open('geoid_lookup.json') as json_file:
        geoid_lookup = json.load(json_file)

    geoid = feats.pop('geoid')
    county_and_puma = geoid_lookup[geoid]
    # del feats["geoid"]
    feats["in.county_and_puma"] = county_and_puma
    return feats


def generate_sample(feats:dict, num_samples:int=100, release:str="2024_1"):
# For each row in dependencies list - check if we have what we need for this column, and if we need it. If so, add a column to the samples df, each of correct distribution according to the existing row
    dep_df = pd.read_csv('appfiles/schema/' + release + '_dependencies.csv')
    column_plan_df = pd.read_csv('column_plan.csv', usecols=['field_name','keep_for_model'])
    # List of columns that the model requires as inputs
    needed_und_cols = column_plan_df.loc[
        (column_plan_df['keep_for_model'] == 'Yes') | 
        (column_plan_df['keep_for_model'] == 'Split')
    ]['field_name'].to_list()

    if release == "2022_1":
        needed_und_cols.remove('in.duct_location')
        needed_und_cols.remove('in.household_has_tribal_persons')
        needed_und_cols.remove('in.clothes_washer_usage_level')
        needed_und_cols.remove('in.clothes_dryer_usage_level')
        needed_und_cols.remove('in.cooking_range_usage_level')
        needed_und_cols.remove('in.refrigerator_usage_level')
        needed_und_cols.remove('in.duct_leakage_and_insulation')
        needed_und_cols.append('in.ducts')

    needed_und_cols.append("in.county_and_puma")
    # Initialize with known parameters
    sample_df_dict = {'bldg_id': list(range(num_samples))}
    for key in feats:
        sample_df_dict[key] = [feats[key]]*num_samples
        # print(f'removing key: {key}')
        needed_und_cols.remove(key)
    
    sample_df = pl.DataFrame(sample_df_dict)
    known_cap_cols  = set(dep_df[dep_df['UnderscoreCase'].isin(feats.keys())]['CapitalCase'].to_list())
    needed_cap_cols = set(dep_df[dep_df['UnderscoreCase'].isin(needed_und_cols)]['CapitalCase'].to_list())
    iter = 0
    while needed_cap_cols:
        iter += 1
        # print(f'iter: {iter}, needed_cap_cols: {needed_cap_cols}')
        # print(f'iter: {iter}, known_cap_cols: {known_cap_cols}')
        for i, field in dep_df['CapitalCase'].items():
            if field not in needed_cap_cols or field in known_cap_cols:
                continue
            dep_str = dep_df.loc[i, 'Dependencies']
            # print(dep_str)
            if pd.isnull(dep_str):
                dependencies = []
            else:
                dependencies = dep_str.split('|')
            
            if any(x not in known_cap_cols for x in dependencies):
                # Set difference - only add dependencies not already known
                needed_cap_cols.update(set(dependencies) - known_cap_cols)
                continue
            else: #All dependencies are already in sample_df
                sample_df = add_col_to_sample(sample_df, field, dependencies, num_samples, release)
                needed_cap_cols.remove(field)
                known_cap_cols.add(field)
        if iter > 500:
            break
    # Can safely remove all columns not needed for model but for dependencies
    model_cols = needed_und_cols + list(feats)
    model_cols.remove("in.county_and_puma")
    state = sample_df.item(0, "in.state")
    gisjoin = sample_df.item(0, "in.county_and_puma").split(", ")[0]
    return clean_sample(sample_df, model_cols), state, gisjoin
    # clean_sample(sample_df, model_cols).write_csv('sample_test.csv')

def clean_sample(sample_df: pl.DataFrame, model_cols:list[str]) -> pl.DataFrame:
    cleaned_df = (sample_df
        .lazy()
        # In.state is in.county before comma
        .drop("in.county")
        .with_columns(
            pl.col("in.county_and_puma").str.split(by=", ").list.first().alias("in.county")
        )
        .select(model_cols)
    )
    
    return cleaned_df.collect()


def add_col_to_sample(sample_df: pl.DataFrame, cap_field: str, cap_dependencies: list[str], num_points: int, release:str="2024_1"):
    directory = 'appfiles/housing_characteristics/' + release
    char_df = pl.scan_parquet(directory + '/' + cap_field + '.parquet')
    options = [col.split('Option=')[1] for col in char_df.columns if col.startswith('Option=')]

    # Easy case where there are no dependencies - only top row of char_df matters
    if not cap_dependencies:
        probs = (char_df
            .lazy()
            .select(pl.selectors.starts_with('Option='))
            .cast(pl.Float64)
            .collect()
            .to_numpy()
        )[0]
        probs /= np.sum(probs)
        samples = np.random.choice(options, p=probs, size=num_points,replace=True).tolist()
        sample_df = sample_df.with_columns(pl.Series(samples).alias(to_underscore_case(cap_field)))
        return sample_df
    
    und_dependencies = [to_underscore_case(d) for d in cap_dependencies]
    
    # Use later to join on
    sample_df_with_deps = (sample_df
        .lazy()
        .with_columns(deps_str=pl.concat_str(und_dependencies, separator="|"))
    )

    sample_col = (sample_df
        .lazy()
        # Will join back up with rest of sample later - only care about what impacts the new column for now
        .select(und_dependencies)
        # Do join on the first dependency (there is guaranteed to be at least one)
        .join(char_df, how='left', left_on=to_underscore_case(cap_dependencies[0]), right_on='Dependency=' + cap_dependencies[0], coalesce=False)
        # Further filter on all remaining dependencies - ideally would have joined on all dependencies but not implemented in polars. Equivalent to cross product and filter by multiple columns.
        .filter(
            # Pl.col(d) is the sample_df column, "Dependency=" is the char_df column" - all dependencies must match exactly
            pl.all_horizontal(
                pl.col(to_underscore_case(d)) == pl.col('Dependency=' + d) for d in cap_dependencies
            )
        )
        .cast({pl.selectors.starts_with('Option='): pl.Float64})
        .with_columns(probs_list=pl.concat_list(pl.selectors.starts_with('Option=')))
        # Normalize probabilities to sum to 1
        .with_columns(pl.col('probs_list').list.eval(pl.element() / pl.element().sum()))
        # Needs to be string rather than list becuase will be joining on it
        .with_columns(deps_str=pl.concat_str(pl.selectors.starts_with('Dependency='), separator="|"))
        # Grouping before calling numpy.random.choice to take advantage of vectorized version of the function - only calling it the minimum number of times, once per unique combo of dependencies
        .group_by("deps_str", "probs_list").len(name='count')
        .with_columns(
            (
                # Ideally this would be map batches for peformance but could not get it to work
                pl.struct(['probs_list', 'count']).map_elements(
                    lambda x: list(np.random.choice(options, p=np.array(x['probs_list'], dtype=float), size=x['count'],replace=True))
                    ,return_dtype=pl.List(pl.String)
                )
            ).alias('choices')
        )
        # Join back up with unique samples df version where dependencies match, so we now have a list of choices at each sample(row)
        .join(sample_df_with_deps, how='inner', on="deps_str", suffix="_sample", coalesce=True)
        # Create an index list column where there are several runs of 0 to number of occurences of that dep combo, restarting for each group. Goal is to choose one of every option generated
        .with_columns(options_idx=pl.int_range(pl.len()).over("deps_str"))
        .with_columns(sample=pl.col("choices").list.get(pl.col("options_idx")))
        .rename({'sample': to_underscore_case(cap_field)})
        .drop('probs_list', 'deps_str', 'count', 'choices', 'options_idx')
    )

    return sample_col.collect()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
