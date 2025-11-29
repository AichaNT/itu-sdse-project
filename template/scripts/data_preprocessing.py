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
