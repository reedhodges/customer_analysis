import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def replace_categorical_outliers(df, cols, threshold):
    """
    Replace values in categorical columns that have a frequency less than a specified threshold with 'Other'.

    Parameters:
    df (DataFrame): The DataFrame to process.
    cols (list of str): The list of categorical column names to check for outliers.
    threshold (int): The frequency threshold under which values are replaced by 'Other'.

    Returns:
    DataFrame: The DataFrame with replaced outlier values in categorical columns.
    """
    for col in cols:
        counts = df[col].value_counts()
        mask = df[col].isin(counts[counts < threshold].index)
        df.loc[mask, col] = 'Other'
    return df

def process_datetime(df, cols):
    """
    Convert columns to datetime and extract the year from each datetime column, dropping the original datetime columns.

    Parameters:
    df (DataFrame): The DataFrame to process.
    cols (list of str): The list of columns to be converted to datetime and processed.

    Returns:
    DataFrame: The DataFrame with processed datetime columns.
    """
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%m-%Y')
        df[col + '_year'] = df[col].dt.year
        df.drop(col, axis=1, inplace=True)
    return df

def preprocess_data(df, categorical_cols, full_date_cols, boolean_cols, numerical_cols):
    df = df.drop_duplicates()
    df.drop(['ID'], axis=1, inplace=True)
    
    df = replace_categorical_outliers(df, categorical_cols, 10)
    df = process_datetime(df, full_date_cols)

    # Creating pipelines for different types of data
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    boolean_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Combining pipelines in a ColumnTransformer
    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('bool', boolean_pipeline, boolean_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

    # Fit and transform the data
    processed_data = preprocessor.fit_transform(df)

    # Getting new column names for categorical columns after OneHotEncoding
    new_categorical_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)

    # Combining all column names
    all_column_names = np.concatenate([new_categorical_cols, boolean_cols, numerical_cols])

    # Creating a new DataFrame with the correct column names
    return pd.DataFrame(processed_data, columns=all_column_names)

def main(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    categorical_cols = ['Education', 'Marital_Status']
    full_date_cols = ['Dt_Customer']
    boolean_cols = ['AcceptedCmp' + str(i) for i in range(1, 6)] + ['Response', 'Complain']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + boolean_cols + full_date_cols + ['ID']]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    train_df = preprocess_data(train_df, categorical_cols, full_date_cols, boolean_cols, numerical_cols)
    test_df = preprocess_data(test_df, categorical_cols, full_date_cols, boolean_cols, numerical_cols)

    return train_df, test_df

# Usage
filepath = 'marketing_campaign.csv'
train_df, test_df = main(filepath)

print(train_df.columns)

