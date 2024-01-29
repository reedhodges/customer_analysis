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

def clean_and_split_data(filepath, categorical_cols, full_date_cols):
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
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop_duplicates()
    df.drop(['ID'], axis=1, inplace=True)
    df = replace_categorical_outliers(df, categorical_cols, 10)
    df = process_datetime(df, full_date_cols)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)
    return train_df, test_df

filepath = 'marketing_campaign.csv'
categorical_cols = ['Education', 'Marital_Status']
full_date_cols = ['Dt_Customer']
train_df, test_df = clean_and_split_data(filepath, categorical_cols, full_date_cols)

boolean_cols = ['AcceptedCmp' + str(i) for i in range(1, 6)] + ['Response', 'Complain']
numerical_cols = [col for col in train_df.columns if col not in categorical_cols + boolean_cols]

cat_imputer = SimpleImputer(strategy='most_frequent')
bool_imputer = SimpleImputer(strategy='constant', fill_value=0)
num_imputer = SimpleImputer(strategy='median')

train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
train_df[boolean_cols] = bool_imputer.fit_transform(train_df[boolean_cols])
train_df[numerical_cols] = num_imputer.fit_transform(train_df[numerical_cols])

test_df[categorical_cols] = cat_imputer.transform(test_df[categorical_cols])
test_df[boolean_cols] = bool_imputer.transform(test_df[boolean_cols])
test_df[numerical_cols] = num_imputer.transform(test_df[numerical_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_columns_train = pd.DataFrame(encoder.fit_transform(train_df[categorical_cols]))
encoded_columns_test = pd.DataFrame(encoder.transform(test_df[categorical_cols]))

# Add encoded columns back to the respective dataframes
train_df = train_df.join(encoded_columns_train).drop(categorical_cols, axis=1)
test_df = test_df.join(encoded_columns_test).drop(categorical_cols, axis=1)

scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

#train_df.to_csv('train.csv', index=False)
#test_df.to_csv('test.csv', index=False)
