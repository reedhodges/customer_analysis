import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def remove_outliers(df, numerical_cols, threshold=1.5):
    """
    Remove outliers from the DataFrame based on the Interquartile Range (IQR) method.

    Parameters:
    df (DataFrame): The DataFrame to process.
    numerical_cols (list of str): List of numerical column names to check for outliers.
    threshold (float): The IQR multiplier to define what is considered an outlier.

    Returns:
    DataFrame: The DataFrame with outliers removed.
    """
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[numerical_cols] < (Q1 - threshold * IQR)) | (df[numerical_cols] > (Q3 + threshold * IQR))
    return df[~mask.any(axis=1)]

def preprocess_data(df, categorical_cols, full_date_cols, boolean_cols, numerical_cols):
    """
    Preprocess the input DataFrame by applying several data transformation steps.

    This function performs operations such as removing duplicates, dropping unnecessary columns,
    replacing categorical outliers, processing datetime columns, and applying transformations 
    like imputation, encoding, and scaling to the data using defined pipelines.

    Parameters:
    df (DataFrame): The input DataFrame to be preprocessed.
    categorical_cols (list of str): List of categorical column names to be processed.
    full_date_cols (list of str): List of datetime column names to be converted to year.
    boolean_cols (list of str): List of boolean column names to be imputed.
    numerical_cols (list of str): List of numerical column names to be imputed and scaled.

    Returns:
    DataFrame: A new DataFrame with the preprocessing applied.
    """
    df = df.drop_duplicates()
    df.drop(['ID'], axis=1, inplace=True)
    
    df = replace_categorical_outliers(df, categorical_cols, 10)
    df = process_datetime(df, full_date_cols)

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('scaler', StandardScaler())
    ])

    boolean_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('bool', boolean_pipeline, boolean_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

    processed_data = preprocessor.fit_transform(df)

    new_categorical_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    all_column_names = np.concatenate([new_categorical_cols, boolean_cols, numerical_cols])

    return pd.DataFrame(processed_data, columns=all_column_names)

def process_file(filepath, split_Q):
    """
    This function reads data from a specified file, identifies different types of columns
    (categorical, datetime, boolean, numerical), and applies preprocessing steps to both 
    training and testing datasets. The data is first split into training and testing sets, 
    and then each set is processed using the preprocess_data function.

    Parameters:
    filepath (str): File path to the dataset to be processed.

    Returns:
    tuple: A tuple containing two DataFrames, (train_df, test_df), 
           which are the training and testing datasets after preprocessing.
    """
    df = pd.read_csv(filepath, delimiter='\t')
    categorical_cols = ['Education', 'Marital_Status']
    full_date_cols = ['Dt_Customer']
    boolean_cols = ['AcceptedCmp' + str(i) for i in range(1, 6)] + ['Response', 'Complain']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + boolean_cols + full_date_cols + ['ID']]

    if split_Q == True:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
        train_df = remove_outliers(train_df, numerical_cols)
        train_df = preprocess_data(train_df, categorical_cols, full_date_cols, boolean_cols, numerical_cols)
        test_df = preprocess_data(test_df, categorical_cols, full_date_cols, boolean_cols, numerical_cols)
        return train_df, test_df
    elif split_Q == False:
        df = remove_outliers(df, numerical_cols)
        df = preprocess_data(df, categorical_cols, full_date_cols, boolean_cols, numerical_cols)
        return df
    else:
        print("Error: split_Q must be True or False")
        return None