# imports
import datetime
import pandas as pd
import os
import mlflow

# Helper functions
def create_dummy_cols(df, col): 
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True) 
    new_df = pd.concat([df, df_dummies], axis=1) 
    new_df = new_df.drop(col, axis=1)
    return new_df

# Re-fetch cont_var and cat_var variables
cat_vars = pd.read_csv("./artifacts/cat_vars_clean.csv")
cont_vars = pd.read_csv("./artifacts/cont_vars_clean.csv")

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
data.to_csv('./artifacts/training_data.csv', index=False)

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
data.to_csv('./artifacts/train_data_gold.csv', index=False)


# Saving constant variables
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv" 
data_version = "00000"
experiment_name = current_date


# Create paths
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)

# Load training data
data = pd.read_csv(data_gold_path)

# Split columns
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

# Create dummy variables for categorical vars
for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

# not finished, might want to either save "preprocessed" data here,
# and then make a separate "split" step, or split and save here - though that does not seem smart.
