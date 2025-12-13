# imports
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

# Helper functions
def describe_numeric_col(x):
    """
    Generate basic descriptive statistics for a numeric pandas Series.

    The function returns a Series containing:
    - Count: Number of non-missing values
    - Missing: Number of missing (NaN) values
    - Mean: Arithmetic mean of the non-missing values
    - Min: Minimum value
    - Max: Maximum value

    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Impute missing values in a pandas Series using a specified strategy.

    For numeric columns (int or float), missing values are imputed using
    either the mean or the median, depending on the selected method.
    For non-numeric columns, missing values are imputed using the mode
    (the most frequent value).

    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


# Re-fetch cont_var and cat_var variables
data = pd.read_csv("./data/interim/clean_data.csv")

# Create categorical data columns
vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]

for col in vars:
    data[col] = data[col].astype("object")

# Separate categorical and continuous columns
cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]

# Handle outliers
cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
outlier_summary = cont_vars.apply(describe_numeric_col).T
outlier_summary.to_csv('./artifacts/outlier_summary.csv')

# Impute missing data
cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")

cont_vars = cont_vars.apply(impute_missing_values)
cont_vars.apply(describe_numeric_col).T

cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
cat_vars = cat_vars.apply(impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T

# Standardise data
scaler_path = "./artifacts/scaler.pkl"

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=scaler_path)

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

# Combine data
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)

# Create data drift artifact
data_columns = list(data.columns)
with open('./artifacts/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)

# Saving data pre-binning
data.to_csv('./data/interim/training_data.csv', index=False)

# Binning object columns
data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)

#Saving gold medallion dataset
data.to_csv('./data/processed/train_data_gold.csv', index=False)