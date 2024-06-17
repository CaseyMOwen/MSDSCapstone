import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

def convert_categorical(X):
    category_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category columns
    for col in category_cols:
        X[col] = X[col].astype('category')
    return X

def seperate_type(X):
    numerical_cols = X.select_dtypes(include=np.number).columns
    category_cols = X.select_dtypes(exclude=np.number).columns
    return X[numerical_cols], X[category_cols]

def fit_scalers(X):
    X_numerical, X_categorical = seperate_type(X)
    scaler = StandardScaler()
    pca = PCA()
    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler.fit(X_numerical)
    pca.fit(X_numerical)
    encoder.fit(X_categorical)
    # X[numerical_cols] = scaled_numerical_data
    return scaler, pca, encoder

def process_data(X: pd.DataFrame, scaler:StandardScaler, pca:PCA, encoder:OneHotEncoder):
    X_numerical, X_categorical = seperate_type(X)
    X_numerical_scaled = scaler.transform(X_numerical)
    X_numerical_pca = pca.transform(X_numerical_scaled)
    X_categorical_encoded = encoder.transform(X_categorical)
    X_processed_array = np.hstack([X_numerical_pca,X_categorical_encoded.toarray()])
    X_processed = pd.concat([pd.Series(X.index), pd.DataFrame(X_processed_array)], axis=1).set_index('Id')
    # X_processed = pd.concat([pd.DataFrame(X_numerical_pca), pd.DataFrame(X_categorical_encoded.toarray())], axis=1)
    return X_processed