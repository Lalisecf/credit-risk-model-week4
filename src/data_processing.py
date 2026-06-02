# ==========================================
# 1. Imports
# ==========================================
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy import sparse


# ==========================================
# 2. Load Data Function
# ==========================================
def load_data(filepath):
    return pd.read_csv(filepath)


# ==========================================
# 3. Aggregate Feature Functions
# ==========================================
def create_aggregate_features(df):

    customer_features = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_transaction_amount=("Amount", "std")
        )
        .reset_index()
    )

    return customer_features


def merge_customer_features(df):

    customer_features = create_aggregate_features(df)

    df = df.merge(
        customer_features,
        on="CustomerId",
        how="left"
    )

    return df


# ==========================================
# 4. Time Feature Extraction
# ==========================================
def extract_time_features(df):

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


# ==========================================
# 5. Feature Lists
# ==========================================
DROP_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "TransactionStartTime"
]

categorical_features = [
    "CurrencyCode",
    "CountryCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy"
]

numerical_features = [
    "Amount",
    "Value",
    "FraudResult",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount"
]


# ==========================================
# 6. Numerical Pipeline
# ==========================================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 7. Categorical Pipeline
# ==========================================
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 8. Column Transformer
# ==========================================
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# 9. Full Pipeline
# ==========================================
full_pipeline = Pipeline([
    ("preprocessor", preprocessor)
])


# ==========================================
# 10. Process Data
# ==========================================
def process_data(df):

    df = extract_time_features(df)

    df = merge_customer_features(df)

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    transformed = full_pipeline.fit_transform(df)

    return transformed


# ==========================================
# 11. Save Processed Data
# ==========================================
def save_processed_data(
    transformed_data,
    filepath
):
    sparse.save_npz(filepath, transformed_data)


# ==========================================
# 12. Main Function
# ==========================================
if __name__ == "__main__":

    df = load_data(
        "data/raw/data.csv"
    )

    processed = process_data(df)

    save_processed_data(
        processed,
        "data/processed/processed_data.npz"
    )

print("Feature engineering completed!")
