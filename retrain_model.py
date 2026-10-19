import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import joblib

# 1. Load the original raw dataset
try:
    df = pd.read_csv("manufacturing_dataset_1000_samples.csv")
except FileNotFoundError:
    print("Error: Place 'manufacturing_dataset_1000_samples.csv' in the same folder as this script!")
    exit(1)

# 2. Re-apply cleaning & preprocessing transformations
df.columns = df.columns.str.strip()

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Timestamp_Hour"] = df["Timestamp"].dt.hour
    df["Timestamp_Month"] = df["Timestamp"].dt.month
    df = df.drop(columns=["Timestamp"])

target = "Parts_Per_Hour"
X = df.drop(columns=[target])
y = df[target]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# 3. Rebuild the pipeline
numeric_transformer_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_scaled = ColumnTransformer(transformers=[
    ("num", numeric_transformer_scaled, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor_scaled),
    ("regressor", Ridge(alpha=1.0))
])

# 4. Train the model locally
print("Training Ridge Regression pipeline on your local environment...")
ridge_model.fit(X, y)

# 5. Export clean model
joblib.dump(ridge_model, "manufacturing_output_model.pkl")
print("Success! A new, compatible 'manufacturing_output_model.pkl' has been generated locally!")
