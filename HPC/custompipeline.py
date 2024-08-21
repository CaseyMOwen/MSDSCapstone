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
import joblib
from datetime import datetime
from pathlib import Path
import modelfunctions as mf
import os
import xgboost as xgb

class CustomPipeline():
    def __init__(self, raw_df, pipeline_path, model_path) -> None:
        self.raw_df = raw_df
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

    def prep_raw(self, df):
        in_cols = [col for col in df.columns if col.startswith('in')]
        out_cols = [col for col in df.columns if col.startswith('out')]
        df['out.electricity.total.energy_consumption.kwh'] = df['out.electricity.total.energy_consumption.kwh'].clip(0,None)

        X, y = df[in_cols], df['out.electricity.total.energy_consumption.kwh']
        X = pf.preprocess_columns(X)
        X = pf.add_tmy3_data(X)
        return X, y

    def fit(self, X, y):
        X, y = self.prep_raw(self.raw_df)
        X_clean = self.pipeline.transform(X)
        dmatrix = xgb.DMatrix(X_clean)

    def predict(self, X):
        