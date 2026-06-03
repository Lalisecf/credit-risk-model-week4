import pandas as pd

from src.data_processing import (
    calculate_rfm
)


def test_rfm_columns():

    df = pd.DataFrame({

        "CustomerId":
        ["C1", "C1", "C2"],

        "TransactionId":
        [1, 2, 3],

        "Value":
        [100, 200, 300],

        "TransactionStartTime":
        [
            "2025-01-01",
            "2025-01-05",
            "2025-01-06"
        ]
    })

    rfm = calculate_rfm(df)

    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_rfm_customer_count():

    df = pd.DataFrame({

        "CustomerId":
        ["C1", "C1", "C2"],

        "TransactionId":
        [1, 2, 3],

        "Value":
        [100, 200, 300],

        "TransactionStartTime":
        [
            "2025-01-01",
            "2025-01-05",
            "2025-01-06"
        ]
    })

    rfm = calculate_rfm(df)

    assert len(rfm) == 2
