from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import xgboost as xgb
import os

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


def set_feature(var, val, input_df):
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
    booster = xgb.Booster()
    # model = xgb.XGBRegressor()
    booster.load_model('appfiles/models/baseline_alabama.json')
    dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
    return booster.predict(dmatrix)



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    feats_df = get_avg_home()
    if request.data:
        feats = request.json  # Expecting JSON input
        for feat in feats:
            feats_df = set_feature(feat, feats[feat], feats_df)
    predictions = predict(feats_df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
