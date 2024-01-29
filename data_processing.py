import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

def impute_categorical_missing_values(df, categorical_cols):
    """
    Impute missing values in categorical columns using the most frequent value.

    Parameters:
    df (DataFrame): The DataFrame to process.
    categorical_cols (list of str): The list of categorical columns for imputation.

    Returns:
    DataFrame: The DataFrame with imputed categorical columns.
    """
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    return df

def impute_numerical_missing_values(df, numerical_cols):
    """
    Impute missing values in numerical columns using the median value.

    Parameters:
    df (DataFrame): The DataFrame to process.
    numerical_cols (list of str): The list of numerical columns for imputation.

    Returns:
    DataFrame: The DataFrame with imputed numerical columns.
    """
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    return df

def clean_data(df, categorical_cols, full_date_cols):
    """
    Perform data cleaning including removing duplicates, replacing categorical outliers, 
    processing datetime columns, and imputing missing values for both categorical and numerical columns.

    Parameters:
    df (DataFrame): The DataFrame to clean.
    categorical_cols (list of str): The list of categorical columns.
    full_date_cols (list of str): The list of full date columns to process.

    Returns:
    DataFrame: The cleaned DataFrame.
    """
    df = df.drop_duplicates()
    df = replace_categorical_outliers(df, categorical_cols, 10)
    df = process_datetime(df, full_date_cols)
    numerical_cols = [col for col in df.columns if col not in categorical_cols + full_date_cols + ['ID']] 
    df = impute_categorical_missing_values(df, categorical_cols)
    df = impute_numerical_missing_values(df, numerical_cols)
    return df

def encode_categorical_variables(df, categorical_cols):
    """
    Encode categorical variables using one-hot encoding.

    Parameters:
    df (DataFrame): The DataFrame to process.
    categorical_cols (list of str): The list of categorical columns to encode.

    Returns:
    DataFrame: The DataFrame with encoded categorical variables.
    """
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df.drop(categorical_cols, axis=1, inplace=True)
    df[encoded_cols] = encoded_data
    return df

def remove_outliers(df, cols_to_check, threshold):
    """
    Remove outliers from specified columns based on Z-score threshold.

    Parameters:
    df (DataFrame): The DataFrame to process.
    cols_to_check (list of str): The list of columns to check for outliers.
    threshold (float): The Z-score threshold to identify outliers.

    Returns:
    DataFrame: The DataFrame with outliers removed.
    """
    for col in cols_to_check:
        z_scores = stats.zscore(df[col])
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < threshold)
        df = df[filtered_entries]
    return df

def scale_data(df, cols_to_scale):
    """
    Scale numerical data columns using Min-Max Scaling.

    Parameters:
    df (DataFrame): The DataFrame to scale.
    cols_to_scale (list of str): The list of columns to scale.

    Returns:
    DataFrame: The scaled DataFrame.
    """
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

def preprocess_data(df, categorical_cols, full_date_cols):
    """
    Execute the full preprocessing pipeline on the DataFrame.

    Parameters:
    df (DataFrame): The original DataFrame.
    categorical_cols (list of str): The list of categorical columns.
    full_date_cols (list of str): The list of full date columns to process.

    Returns:
    DataFrame: The fully preprocessed DataFrame.
    """
    df = df.copy()
    df = clean_data(df, categorical_cols, full_date_cols)
    df = encode_categorical_variables(df, categorical_cols)
    df = df.astype(int)
    cols_to_check_for_outliers = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    df = remove_outliers(df, cols_to_check_for_outliers, 4)
    cols_to_scale = [col for col in df.columns if col not in ['ID']]
    df = scale_data(df, cols_to_scale)
    return df

filepath = 'marketing_campaign.csv'
data = pd.read_csv(filepath, delimiter='\t')

categorical_cols = ['Education', 'Marital_Status']
full_date_cols = ['Dt_Customer']

cleaned_data = preprocess_data(data, categorical_cols, full_date_cols)

# save to csv
cleaned_data.to_csv('cleaned_data.csv', index=False)
