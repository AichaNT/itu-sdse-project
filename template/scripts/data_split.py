from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Helper functions
def create_dummy_cols(df, col): 
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True) 
    new_df = pd.concat([df, df_dummies], axis=1) 
    new_df = new_df.drop(col, axis=1)
    return new_df

# Saving constant variables
data_gold_path = "./artifacts/train_data_gold.csv" 

# Create paths
os.makedirs("artifacts", exist_ok=True)

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

# Splitting data
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y)


train_df = X_train.copy()
train_df["lead_indicator"] = y_train

test_df = X_test.copy()
test_df["lead_indicator"] = y_test

train_df.to_csv("./artifacts/train.csv", index=False)
test_df.to_csv("./artifacts/test.csv", index=False)