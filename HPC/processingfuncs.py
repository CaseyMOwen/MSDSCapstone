import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor
import os.path
import random

'''
Automatically build ordinal mapping format from a string delimited with "|" (csv safe character)
'''
def get_ordinal_mapping(text:str):
    mapping = {}
    split_array = text.split("|")
    for i, item in enumerate(split_array):
        if i % 2 == 1:
            mapping[split_array[i-1]] = int(split_array[i])
    return mapping


def drop_ignored_columns(X: pd.DataFrame):
    column_plan_df = pd.read_csv('column_plan.csv')
    to_drop = []
    for index, row in column_plan_df.iterrows():
        field = row['field_name']
        keep = row['keep_for_model']
        if keep == "No" and field in X:
            to_drop.append(field)
    X = X.drop(columns=to_drop)
    return X

def str_to_int(text:str):
    if text == "None":
        return 0
    else:
        return int(text)

def split_columns(X:pd.DataFrame):
    # Split possible outcomes by comma, leakage and insulation are two features
    X[['in.duct_leakage','in.duct_insulation']] = X['in.duct_leakage_and_insulation'].str.split(', ',expand=True)
    X = X.drop(columns=['in.duct_leakage_and_insulation'])

    # Columns that are integers with "None" as 0
    X['in.geometry_building_number_units_mf'] = X['in.geometry_building_number_units_mf'].apply(str_to_int)
    X['in.geometry_building_number_units_sfa'] = X['in.geometry_building_number_units_sfa'].apply(str_to_int)

    # Convert string int columns to numeric
    X['in.geometry_building_number_units_mf'] = pd.to_numeric(X['in.geometry_building_number_units_mf'])
    X['in.geometry_building_number_units_sfa'] = pd.to_numeric(X['in.geometry_building_number_units_sfa'])
    X['in.geometry_stories'] = pd.to_numeric(X['in.geometry_stories'])
    X['in.bedrooms'] = pd.to_numeric(X['in.bedrooms'])

    # Split "insulation_wall into two columns - insulation rating, and wall material"
    X['in.wall_type'] = X['in.insulation_wall'].str.split(',').str[0]
    X['in.wall_insulation_rating'] = X['in.insulation_wall'].str.split(', ').str[-1]
    X = X.drop(columns=['in.insulation_wall'])
    
    # Retain only heating type
    X['in.hvac_heating_efficiency'] = X['in.hvac_heating_efficiency'].str.split(',').str[0]

    # Convert compass direction to two-dimensional vector for each feature
    X['in.orientation_northness'] = X['in.orientation']
    X['in.orientation_eastness'] = X['in.orientation']
    X['in.pv_orientation_northness'] = X['in.pv_orientation']
    X['in.pv_orientation_eastness'] = X['in.pv_orientation']
    X = X.drop(columns=['in.orientation', 'in.pv_orientation'])
    return X

def convert_categorical(X: pd.DataFrame):
    category_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category 
    pd.options.mode.chained_assignment = None  # default='warn'
    for col in category_cols:
        X[col] = X[col].astype('category')
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return X


def preprocess_columns(X: pd.DataFrame):
    # X = convert_categorical(X)
    X = drop_ignored_columns(X)
    X = split_columns(X)
    X = convert_categorical(X)
    return X

def get_custom_mappings():
    # Add encodings to columns that were just split

    custom_binary_cols = ['in.wall_type', 'in.hvac_heating_efficiency']

    # Manual Oridinal mappings from columns that were split above
    # All other ordinal mapping schemes are in "column_plan.csv"
    custom_ordinal_mappings = [
        {
            'col':'in.duct_leakage',
            'mapping':{r'20% Leakage': 20, r'30% Leakage': 30, r'0% Leakage': 0, r'10% Leakage': 10, 'None': -10}
        },
        {
            'col':'in.duct_insulation',
            'mapping':{'Uninsulated': 0, 'R-4': 4, 'R-6': 6, 'R-8': 8, None: -10}},
        {
            'col':'in.wall_insulation_rating',
            'mapping':{'Uninsulated': 0, 'R-11': 11, 'R-15': 15, 'R-19': 19, 'R-7': 7,}
        },
        {
            'col':'in.orientation_northness',
            'mapping':{'North': 4, 'Southwest': 1, 'West': 2, 'Northeast': 3, 'South': 0, 'Northwest': 3, 'East': 2, 'Southeast': 1, "None": 2}
        },
        {
            'col':'in.orientation_eastness',
            'mapping':{'North': 2, 'Southwest': 1, 'West': 0, 'Northeast': 3, 'South': 2, 'Northwest': 1, 'East': 4, 'Southeast': 3, "None": 2}
        },
        {
            'col':'in.pv_orientation_northness',
            'mapping':{'North': 4, 'Southwest': 1, 'West': 2, 'Northeast': 3, 'South': 0, 'Northwest': 3, 'East': 2, 'Southeast': 1, "None": 2}
        },
        {
            'col':'in.pv_orientation_eastness',
            'mapping':{'North': 2, 'Southwest': 1, 'West': 0, 'Northeast': 3, 'South': 2, 'Northwest': 1, 'East': 4, 'Southeast': 3, "None": 2}
        }
    ]
    return custom_binary_cols, custom_ordinal_mappings

def build_column_transformer():
    column_plan_df = pd.read_csv('column_plan.csv')
    custom_binary_cols, custom_ordinal_mappings = get_custom_mappings()

    # Binary
    binary_fields = column_plan_df[column_plan_df['encoder'] == 'Binary']['field_name'].to_list()
    binary_fields += custom_binary_cols
    binary_encoder = ce.BinaryEncoder(cols=binary_fields)

    # Ordinal
    full_mapping = []
    for index, row in column_plan_df.iterrows():
        if row['encoder'] != 'Ordinal':
            continue
        mapping_dict = {}
        mapping_dict['col'] = row['field_name']
        mapping_dict['mapping'] = get_ordinal_mapping(row['label_encoder_dict'])
        full_mapping.append(mapping_dict)

    full_mapping += custom_ordinal_mappings
    ordinal_fields = []
    for mapping in full_mapping:
        ordinal_fields.append(mapping['col'])
    ordinal_encoder = ce.OrdinalEncoder(cols=ordinal_fields, mapping=full_mapping)

    # Catboost
    catboost_fields = column_plan_df[column_plan_df['encoder'] == 'CatBoost']['field_name'].to_list()
    catboost_encoder = ce.CatBoostEncoder(cols=catboost_fields)

    preprocessor = ColumnTransformer(
    transformers=[
        ('binary', binary_encoder, binary_fields),
        ('ordinal', ordinal_encoder, ordinal_fields),
        ('catboost', catboost_encoder, catboost_fields),
    ],
    remainder='passthrough'
)
    return preprocessor

def create_hyperparemeter_df(hyperparameters: dict) -> pd.DataFrame:
    '''
    Purpose: 
        Creates a hyperparameter dataframe, that includes all possible combinations (the cross product) of the given hyperparameters
    Inputs: 
        -hyperparameters: A dict of all possible values of each hyperparameter to take the cross product of. Must include keys "learning_rate", "max_depth", and "n_estimators", where the values are lists of all possible values that parameter should take on
    Outputs: 
        -cross_product_df: A dataframe where each row is a possible combo of hyperparameters
    '''
    cross_product_df = None
    for key in hyperparameters:
        hyp_df = pd.DataFrame({key:hyperparameters[key]})
        if cross_product_df is None:
            cross_product_df = hyp_df
        else:
            cross_product_df = pd.merge(cross_product_df, hyp_df, how='cross')
    return cross_product_df


# def write_results_df():
#     # results_df = 
#     pass

def random_search(hyperparameters: dict, X: pd.DataFrame, y: pd.DataFrame, cv:int, n_iter:int, results_path:str):
    hyp_df = create_hyperparemeter_df(hyperparameters).sample(n=n_iter)
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = None
    for index, row in hyp_df.iterrows():
        if results_df is not None and any((
                (results_df['learning_rate'] == row['learning_rate']) &
                (results_df['max_depth'] == row['max_depth']) &
                (results_df['n_estimators'] == row['n_estimators']))):
            continue
        else:
            print(f"training model for with learning learning_rate: {row['learning_rate']}, max_depth: {row['max_depth']}, n_estimators: {row['n_estimators']}")
            estimator = XGBRegressor(
                objective = 'reg:squarederror',
                tree_method = 'hist',
                device="cuda",
                verbosity=0,
                learning_rate=row['learning_rate'],
                max_depth=int(row['max_depth']),
                n_estimators=int(row['n_estimators'])
            )
            cv_results = cross_validate(estimator, X, y, cv=cv, scoring=('r2', 'neg_root_mean_squared_error'), return_train_score=True)
            # print(cv_results.keys())
            new_row = pd.DataFrame({
                'learning_rate': row['learning_rate'],
                'max_depth': int(row['max_depth']),
                'n_estimators': int(row['n_estimators']),
                'mean_fit_time': np.mean(cv_results['fit_time']),
                'std_fit_time': np.std(cv_results['fit_time']),
                'mean_score_time': np.mean(cv_results['score_time']),
                'std_score_time': np.std(cv_results['score_time']),
                'mean_test_r2': np.mean(cv_results['test_r2']),
                'test_r2': np.std(cv_results['test_r2']),
                'mean_train_r2': np.mean(cv_results['train_r2']),
                'train_r2': np.std(cv_results['train_r2']),
                'mean_test_neg_root_mean_squared_error': np.mean(cv_results['test_neg_root_mean_squared_error']),
                'std_test_neg_root_mean_squared_error': np.std(cv_results['test_neg_root_mean_squared_error']),
                'mean_train_neg_root_mean_squared_error': np.mean(cv_results['train_neg_root_mean_squared_error']),
                'std_train_neg_root_mean_squared_error': np.std(cv_results['train_neg_root_mean_squared_error']),
            }, index=[0])
            header = not os.path.exists(results_path)
            new_row.to_csv(results_path, mode='a', index=False, header=header)

