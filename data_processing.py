import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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
    df.drop(['ID'], axis=1, inplace=True)
    df = replace_categorical_outliers(df, categorical_cols, 10)
    df = process_datetime(df, full_date_cols)
    return df

def impute_categorical_missing_values(df, categorical_cols):
    """
    Impute missing values in categorical columns using the most frequent value.

    Parameters:
    df (DataFrame): The DataFrame to process.
    categorical_cols (list of str): The list of categorical columns for imputation.

    Returns:
    DataFrame: The DataFrame with imputed categorical columns.
    cat_imputer: The fitted imputer object.
    """
    cat_imputer = SimpleImputer(strategy='most_frequent').fit(df[categorical_cols])
    df[categorical_cols] = cat_imputer.transform(df[categorical_cols])
    return df, cat_imputer

def impute_numerical_missing_values(df, numerical_cols):
    """
    Impute missing values in numerical columns using the median value.

    Parameters:
    df (DataFrame): The DataFrame to process.
    numerical_cols (list of str): The list of numerical columns for imputation.

    Returns:
    DataFrame: The DataFrame with imputed numerical columns.
    num_imputer: The fitted imputer object.
    """
    num_imputer = SimpleImputer(strategy='median').fit(df[numerical_cols])
    df[numerical_cols] = num_imputer.transform(df[numerical_cols])
    return df, num_imputer

def encode_categorical_variables(df, categorical_cols):
    """
    Encode categorical variables using one-hot encoding.

    Parameters:
    df (DataFrame): The DataFrame to process.
    categorical_cols (list of str): The list of categorical columns to encode.

    Returns:
    DataFrame: The DataFrame with encoded categorical variables.
    encoder: The fitted encoder object.
    """
    encoder = OneHotEncoder(sparse_output=False).fit(df[categorical_cols])
    encoded_data = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(categorical_cols, axis=1, inplace=True)
    return df, encoder, encoded_cols

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
    scaler: The fitted scaler object.
    """
    scaler = MinMaxScaler().fit(df[cols_to_scale])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df, scaler

def preprocess_data(filepath, categorical_cols, full_date_cols, cols_to_check_for_outliers):
    """
    Execute the full preprocessing pipeline on the DataFrame.

    Parameters:
    df (DataFrame): The original DataFrame.
    categorical_cols (list of str): The list of categorical columns.
    full_date_cols (list of str): The list of full date columns to process.

    Returns:
    DataFrame: The fully preprocessed DataFrame.
    """
    df = pd.read_csv(filepath, delimiter='\t')
    df = clean_data(df, categorical_cols, full_date_cols)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # save train_df to csv
    #train_df.to_csv('train.csv', index=False)
    #train_df = remove_outliers(train_df, cols_to_check_for_outliers, 3)

    train_df, cat_imputer = impute_categorical_missing_values(train_df, categorical_cols)
    test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])

    train_df, encoder, encoded_cols = encode_categorical_variables(train_df, categorical_cols)
    encoded_test_data = encoder.transform(test_df[categorical_cols])
    encoded_test_df = pd.DataFrame(encoded_test_data, columns=encoded_cols, index=test_df.index)
    test_df = pd.concat([test_df, encoded_test_df], axis=1)
    test_df.drop(categorical_cols, axis=1, inplace=True)

    numerical_cols = train_df.select_dtypes(include=['number']).columns
    train_df, num_imputer = impute_numerical_missing_values(train_df, numerical_cols)
    test_df[numerical_cols] = num_imputer.transform(test_df[numerical_cols])

    train_df, scaler = scale_data(train_df, numerical_cols)
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    return train_df, test_df

categorical_cols = ['Education', 'Marital_Status']
full_date_cols = ['Dt_Customer']
cols_to_check_for_outliers = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

filepath = 'marketing_campaign.csv'
train_df, test_df = preprocess_data(filepath, categorical_cols, full_date_cols, cols_to_check_for_outliers)
print(train_df.shape)
print(test_df.shape)
# save to csv
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)