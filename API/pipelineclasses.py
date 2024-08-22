import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.compose import ColumnTransformer

class Preprocessing():
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        # self.column_plan_df = pd.read_csv('column_plan.csv', usecols=['field_name','keep_for_model'])
        pass


    def transform(self, X):
        X = self.split_columns(X)
        # X = self.convert_categorical(X)
        return X
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    '''
    def drop_ignored_columns(self, X: pd.DataFrame):
        # column_plan_df = pd.read_csv('column_plan.csv')
        to_drop = []
        for index, row in self.column_plan_df.iterrows():
            field = row['field_name']
            keep = row['keep_for_model']
            if keep == "No" and field in X:
                to_drop.append(field)
        X = X.drop(columns=to_drop)
        return X
    '''
    def str_to_int(self, text:str):
        if text == "None":
            return 0
        else:
            return int(text)

    def split_columns(self, X:pd.DataFrame):
        # Split possible outcomes by comma, leakage and insulation are two features
        to_drop = []
        X[['in.duct_leakage','in.duct_insulation']] = X['in.duct_leakage_and_insulation'].str.split(', ',expand=True)
        X['in.duct_insulation'] = X['in.duct_insulation'].fillna('None')

        to_drop.append('in.duct_leakage_and_insulation')
        # X = X.drop(columns=['in.duct_leakage_and_insulation'])

        # Columns that are integers with "None" as 0
        X['in.geometry_building_number_units_mf'] = X['in.geometry_building_number_units_mf'].apply(self.str_to_int)
        X['in.geometry_building_number_units_sfa'] = X['in.geometry_building_number_units_sfa'].apply(self.str_to_int)

        # Convert string int columns to numeric
        X['in.geometry_building_number_units_mf'] = pd.to_numeric(X['in.geometry_building_number_units_mf'])
        X['in.geometry_building_number_units_sfa'] = pd.to_numeric(X['in.geometry_building_number_units_sfa'])
        X['in.geometry_stories'] = pd.to_numeric(X['in.geometry_stories'])
        X['in.bedrooms'] = pd.to_numeric(X['in.bedrooms'])

        # Split "insulation_wall into two columns - insulation rating, and wall material"
        X['in.wall_type'] = X['in.insulation_wall'].str.split(',').str[0]
        X['in.wall_insulation_rating'] = X['in.insulation_wall'].str.split(', ').str[-1]
        to_drop.append('in.insulation_wall')
        # X = X.drop(columns=['in.insulation_wall'])
        
        # Retain only heating type
        X['in.hvac_heating_efficiency'] = X['in.hvac_heating_efficiency'].str.split(',').str[0]

        # Convert compass direction to two-dimensional vector for each feature
        X['in.orientation_northness'] = X['in.orientation']
        X['in.orientation_eastness'] = X['in.orientation']
        X['in.pv_orientation_northness'] = X['in.pv_orientation']
        X['in.pv_orientation_eastness'] = X['in.pv_orientation']
        to_drop.append('in.orientation')
        to_drop.append('in.pv_orientation')
        X = X.drop(columns=to_drop)
        return X

    '''
    def convert_categorical(self, X: pd.DataFrame):
        category_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Convert to Pandas category 
        pd.options.mode.chained_assignment = None  # default='warn'
        for col in category_cols:
            X[col] = X[col].astype('category')
        pd.options.mode.chained_assignment = 'warn'  # default='warn'
        return X
    '''

    # def preprocess_columns(self, X: pd.DataFrame):
    #     # X = convert_categorical(X)
    #     X = self.drop_ignored_columns(X)
    #     X = self.split_columns(X)
    #     X = self.convert_categorical(X)
    #     return X
    
class Encoding():
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        self.column_plan_df = pd.read_csv('column_plan.csv').filter(['field_name','encoder', 'label_encoder_dict'])
        self.preprocesser = self.build_column_transformer().set_output(transform='pandas')
        self.preprocesser.fit(X, y)


    def transform(self, X):
        return self.preprocesser.transform(X)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_custom_mappings(self):
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
    
    '''
    Automatically build ordinal mapping format from a string delimited with "|" (csv safe character)
    '''
    def get_ordinal_mapping(self, text:str):
        mapping = {}
        split_array = text.split("|")
        for i, item in enumerate(split_array):
            if i % 2 == 1:
                mapping[split_array[i-1]] = int(split_array[i])
        return mapping
    
    def build_column_transformer(self):
        # column_plan_df = pd.read_csv('column_plan.csv')
        custom_binary_cols, custom_ordinal_mappings = self.get_custom_mappings()

        # Binary
        binary_fields = self.column_plan_df[self.column_plan_df['encoder'] == 'Binary']['field_name'].to_list()
        binary_fields += custom_binary_cols
        binary_encoder = ce.BinaryEncoder(cols=binary_fields, drop_invariant=True)

        # Ordinal
        full_mapping = []
        for index, row in self.column_plan_df.iterrows():
            if row['encoder'] != 'Ordinal':
                continue
            mapping_dict = {}
            mapping_dict['col'] = row['field_name']
            mapping_dict['mapping'] = self.get_ordinal_mapping(row['label_encoder_dict'])
            full_mapping.append(mapping_dict)

        full_mapping += custom_ordinal_mappings
        ordinal_fields = []
        for mapping in full_mapping:
            ordinal_fields.append(mapping['col'])
        ordinal_encoder = ce.OrdinalEncoder(cols=ordinal_fields, mapping=full_mapping, drop_invariant=True)

        # Catboost
        catboost_fields = self.column_plan_df[self.column_plan_df['encoder'] == 'CatBoost']['field_name'].to_list()
        catboost_encoder = ce.CatBoostEncoder(cols=catboost_fields, drop_invariant=True)

        preprocessor = ColumnTransformer(
        transformers=[
            ('binary', binary_encoder, binary_fields),
            ('ordinal', ordinal_encoder, ordinal_fields),
            ('catboost', catboost_encoder, catboost_fields),
        ],
        remainder='passthrough'
    )
        return preprocessor
    
class AddWeatherData():
    def __init__(self, year_range='current', state=None, gisjoin=None) -> None:
        self.params = {
            "year_range": year_range,
            "state": state,
            "gisjoin": gisjoin,
        }
    def fit(self, X, y):
        pass

    def transform(self, X):
        if self.params["year_range"] == 'current':
            return self.add_tmy3_data(X)
        else:
            state, gisjoin, year_range = self.params["state"], self.params["gisjoin"], self.params["year_range"]
            return self.add_ftmy3_data(X, state, gisjoin, year_range)

        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def add_tmy3_data(self, X:pd.DataFrame):
        tmy3_df = pd.read_csv('TMY3_aggregates.csv', index_col='gisjoin')
        combined = X.join(tmy3_df, on='in.county', how='left')
        # combined.set_index('bldg_id')
        # print(combined)
        # combined.to_csv('test_join.csv')
        combined = combined.drop(columns=['state', 'county', 'in.county'])
        return combined

    def add_ftmy3_data(self, X:pd.DataFrame, state:str, gisjoin:str, year_range:str):
        ftmy3_df = pd.read_csv(f'appfiles/fTMY3_aggregates/{state}/{gisjoin}/{year_range}.csv', index_col='gisjoin')
        combined = X.join(ftmy3_df, on='in.county', how='left')
        combined = combined.drop(columns=['state', 'county', 'year range', 'in.county'])
        return combined
    
    def set_params(self, **parameters):
        for key, value in parameters.items():
            self.params[key] = value