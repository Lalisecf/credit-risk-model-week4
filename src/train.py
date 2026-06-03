# ==========================================
# train.py
# Credit Risk Model Training + MLflow
# ==========================================

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_processing import (
    preprocessor
)

# ==========================================
# Directories
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Load Processed Dataset
# ==========================================

X = pd.read_csv(
    "data/processed/processed_dataset.csv"
)

y = pd.read_csv(
    "data/processed/target.csv"
).squeeze()

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MLflow
# ==========================================

mlflow.set_experiment(
    "credit_risk_probability_model"
)

# ==========================================
# Evaluation Function
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    return {

        "accuracy":
        accuracy_score(
            y_test,
            y_pred
        ),

        "precision":
        precision_score(
            y_test,
            y_pred
        ),

        "recall":
        recall_score(
            y_test,
            y_pred
        ),

        "f1_score":
        f1_score(
            y_test,
            y_pred
        ),

        "roc_auc":
        roc_auc_score(
            y_test,
            y_prob
        )
    }

# ==========================================
# Logistic Regression
# ==========================================

with mlflow.start_run(
    run_name="LogisticRegression"
):

    lr_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            )
        ]
    )

    grid_lr = GridSearchCV(
        lr_pipeline,
        {
            "classifier__C": [
                0.01,
                0.1,
                1,
                10
            ]
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_lr.fit(
        X_train,
        y_train
    )

    best_lr = (
        grid_lr.best_estimator_
    )

    metrics = evaluate_model(
        best_lr,
        X_test,
        y_test
    )

    mlflow.log_params(
        grid_lr.best_params_
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.sklearn.log_model(
        best_lr,
        "model"
    )

    print("\nLogistic Regression")
    print(metrics)

# ==========================================
# Random Forest
# ==========================================

with mlflow.start_run(
    run_name="RandomForest"
):

    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            )
        ]
    )

    grid_rf = GridSearchCV(
        rf_pipeline,
        {
            "classifier__n_estimators": [
                100,
                200
            ],
            "classifier__max_depth": [
                5,
                10,
                None
            ]
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_rf.fit(
        X_train,
        y_train
    )

    best_rf = (
        grid_rf.best_estimator_
    )

    metrics = evaluate_model(
        best_rf,
        X_test,
        y_test
    )

    mlflow.log_params(
        grid_rf.best_params_
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.sklearn.log_model(
        best_rf,
        "model"
    )

    print("\nRandom Forest")
    print(metrics)

# ==========================================
# Compare Models
# ==========================================

lr_auc = evaluate_model(
    best_lr,
    X_test,
    y_test
)["roc_auc"]

rf_auc = evaluate_model(
    best_rf,
    X_test,
    y_test
)["roc_auc"]

if rf_auc > lr_auc:

    best_model = best_rf
    best_name = "RandomForest"

else:

    best_model = best_lr
    best_name = "LogisticRegression"

print(
    f"\nBest Model: {best_name}"
)

# ==========================================
# Save Best Model
# ==========================================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print(
    "\nModel Saved Successfully"
)

# ==========================================
# Register Model
# ==========================================

with mlflow.start_run(
    run_name="BestModelRegistry"
):

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="CreditRiskModel"
    )