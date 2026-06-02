# ==========================================
# data_processing.py
# Task 3 + Task 4
# Credit Risk Probability Model
# ==========================================

import pandas as pd
import joblib

from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# ==========================================
# 1. Load Data
# ==========================================

def load_data(filepath):
    return pd.read_csv(filepath)


# ==========================================
# 2. Custom Transformer:
# Time Features
# ==========================================

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["transaction_hour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["transaction_day"] = (
            X["TransactionStartTime"].dt.day
        )

        X["transaction_month"] = (
            X["TransactionStartTime"].dt.month
        )

        X["transaction_year"] = (
            X["TransactionStartTime"].dt.year
        )

        return X


# ==========================================
# 3. Custom Transformer:
# Aggregate Customer Features
# ==========================================

class AggregateFeatureTransformer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):

        self.customer_features_ = (
            X.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                avg_transaction_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_transaction_amount=("Amount", "std")
            )
            .reset_index()
        )

        self.customer_features_[
            "std_transaction_amount"
        ] = (
            self.customer_features_[
                "std_transaction_amount"
            ].fillna(0)
        )

        return self

    def transform(self, X):

        X = X.copy()

        X = X.merge(
            self.customer_features_,
            on="CustomerId",
            how="left"
        )

        return X


# ==========================================
# 4. RFM Calculation
# ==========================================

def calculate_rfm(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )

    return rfm


# ==========================================
# 5. Create Proxy Target
# ==========================================

def create_proxy_target(df):

    rfm = calculate_rfm(df)

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            ["Recency",
             "Frequency",
             "Monetary"]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = (
        rfm.groupby("Cluster")
        .agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        })
    )

    cluster_summary["RiskScore"] = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"]
    )

    high_risk_cluster = (
        cluster_summary["RiskScore"]
        .idxmax()
    )

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    return (
        rfm[
            ["CustomerId",
             "is_high_risk"]
        ],
        kmeans
    )


# ==========================================
# 6. Feature Lists
# ==========================================

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
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount"
]

DROP_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "TransactionStartTime",
    "CustomerId"
]


# ==========================================
# 7. Numerical Pipeline
# ==========================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ==========================================
# 8. Categorical Pipeline
# ==========================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ==========================================
# 9. Column Transformer
# ==========================================

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numerical_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ==========================================
# 10. Full Pipeline
# ==========================================

full_pipeline = Pipeline([
    (
        "time_features",
        TimeFeatureExtractor()
    ),
    (
        "aggregate_features",
        AggregateFeatureTransformer()
    ),
    (
        "preprocessor",
        preprocessor
    )
])


# ==========================================
# 11. Process Data
# ==========================================

def process_data(df):

    target_df, kmeans = create_proxy_target(df)

    df = df.merge(
        target_df,
        on="CustomerId",
        how="left"
    )

    audit_df = df.copy()

    y = df["is_high_risk"]

    X = full_pipeline.fit_transform(
        df.drop(
            columns=["is_high_risk"]
        )
    )
    
    feature_names = (
    full_pipeline.named_steps[
        "preprocessor"
    ]
    .get_feature_names_out()
    )

    X_df = pd.DataFrame(
        X.toarray(),
        columns=feature_names
    )

    processed_df = (
        full_pipeline.named_steps[
            "time_features"
        ]
        .transform(df)
    )

    processed_df = (
        full_pipeline.named_steps[
            "aggregate_features"
        ]
        .transform(processed_df)
    )

    processed_df = processed_df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    return (
    X,
    X_df,
    y,
    processed_df,
    target_df,
    audit_df,
    kmeans
)


# ==========================================
# 12. Save Sparse Matrix
# ==========================================

def save_processed_data(
    transformed_data,
    filepath
):

    sparse.save_npz(
        filepath,
        transformed_data
    )


# ==========================================
# 13. Main
# ==========================================

if __name__ == "__main__":

    df = load_data(
        "data/raw/data.csv"
    )

    (
        X,
        X_df,
        y,
        processed_df,
        target_df,
        audit_df,
        kmeans
    ) = process_data(df)

    # Save transformed matrix
    save_processed_data(
        X,
        "data/processed/processed_data.npz"
    )

    # Save target
    y.to_csv(
        "data/processed/target.csv",
        index=False
    )

    # Save processed dataframe
    processed_df.to_csv(
        "data/processed/processed_dataset.csv",
        index=False
    )

    # Save customer risk labels
    target_df.to_csv(
        "data/processed/customer_risk_labels.csv",
        index=False
    )

    # Save audit dataset
    audit_df.to_csv(
        "data/processed/processed_dataset_with_target.csv",
        index=False
    )

    # Save target distribution
    y.value_counts().to_csv(
        "data/processed/risk_distribution.csv"
    )

    X_df.to_csv(
    "data/processed/model_ready_features.csv",
    index=False
    )    

    # Save preprocessing pipeline
    joblib.dump(
        full_pipeline,
        "models/preprocessing_pipeline.pkl"
    )

    # Save KMeans model
    joblib.dump(
        kmeans,
        "models/kmeans_rfm.pkl"
    )
    

    print("Feature engineering completed successfully!")
    print("\nRisk Distribution:")
    print(y.value_counts())