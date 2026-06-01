# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end Credit Risk Probability Model for Bati Bank using transaction data from the Xente eCommerce platform.

The objective is to identify high-risk customers, estimate default probability, generate credit scores, and support Buy-Now-Pay-Later (BNPL) lending decisions using alternative financial behavior data.

Since the dataset does not contain actual loan repayment outcomes, a proxy target variable will be constructed using customer behavioral patterns derived from transaction history.

---

# Project Structure

```text
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
├── notebooks/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
```

---

# Environment Setup

## Clone Repository

```bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run EDA Notebook

```bash
jupyter notebook notebooks/
```

Open and execute the EDA notebook to reproduce all exploratory analysis results.

## Train the Model

```bash
python src/train.py
```

## Run Tests

```bash
pytest tests/
```

## Run CI Checks Locally

```bash
flake8 src tests
pytest
```

---

# Configuration

Project configuration is managed through:

* requirements.txt for dependencies
* environment variables for secrets and deployment settings
* configurable file paths stored within project modules

No sensitive credentials are stored in the repository.

---

# Task 1: Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Accord requires financial institutions to maintain transparent, reliable, and auditable credit risk assessment processes. Credit scoring models directly influence lending decisions, capital allocation, regulatory reporting, and overall financial stability.

Because these decisions affect both customers and regulatory compliance, models must be:

* Interpretable
* Auditable
* Reproducible
* Well documented

Interpretability enables risk analysts and regulators to understand why a customer was classified as high-risk or low-risk.

Documentation supports:

* Regulatory audits
* Model validation
* Performance monitoring
* Ongoing model governance

---

## 2. Why a Proxy Variable Is Necessary

The dataset contains transaction records but does not include historical repayment outcomes or default labels.

Since supervised machine learning requires a target variable, a proxy measure of risk must be created.

The proposed approach uses customer behavior based on:

* Recency
* Frequency
* Monetary Value (RFM)

Customers exhibiting low engagement, low transaction frequency, and low transaction value may be classified as higher-risk customers.

### Risks of Proxy-Based Prediction

* Proxy labels may not accurately represent actual default behavior.
* Inactive customers may still be financially reliable.
* Models may learn engagement patterns rather than repayment risk.
* Lending decisions may be influenced by imperfect assumptions.

Therefore, the proxy target should be treated as an approximation rather than ground truth.

---

## 3. Interpretable vs High-Performance Models

### Logistic Regression with WoE

#### Advantages

* Highly interpretable
* Easy to explain to regulators
* Stable and transparent
* Supports scorecard development
* Easier monitoring and governance

#### Limitations

* Limited ability to model nonlinear relationships
* Potentially lower predictive performance

### Gradient Boosting Models

#### Advantages

* Higher predictive accuracy
* Captures nonlinear interactions
* Handles complex customer behavior patterns

#### Limitations

* Less transparent
* More difficult to explain
* Increased regulatory scrutiny

### Recommended Approach

Both Logistic Regression and Gradient Boosting models will be evaluated.

Model selection will balance:

* Predictive performance
* Interpretability
* Regulatory compliance
* Operational maintainability

SHAP-based explanations will be explored for advanced models.

---

# Model Governance and Monitoring

To align with Basel II expectations, the following governance framework will be applied.

## Backtesting

Model performance will be reviewed monthly by comparing predicted risk levels against actual customer repayment behavior once repayment data becomes available.

## Independent Validation

Independent model validation should occur at least annually to evaluate:

* Predictive performance
* Stability
* Fairness
* Regulatory compliance

## Model Change Management

Any changes to:

* Feature definitions
* Proxy target construction
* Model algorithms
* Hyperparameters

must be documented, validated, tested, and formally approved before deployment.

## Performance Monitoring

The following metrics will be monitored:

* AUC-ROC
* Precision
* Recall
* F1 Score
* Population Stability Index (PSI)

This helps detect model drift and performance degradation.

---

## Proxy Failure Scenarios

| Scenario                                     | Risk                                | Mitigation                   |
| -------------------------------------------- | ----------------------------------- | ---------------------------- |
| Active customer unexpectedly defaults        | Proxy misses true credit risk       | Conservative credit limits   |
| Low activity customer is financially healthy | False high-risk classification      | Manual review process        |
| Seasonal transaction behavior                | Incorrect risk assignment           | Periodic model recalibration |
| New customer with limited history            | Insufficient behavioral information | Lower initial lending limits |

---

# Task 2: Exploratory Data Analysis (EDA)

## Objective

The EDA phase was conducted to:

* Understand dataset structure
* Assess data quality
* Identify missing values
* Detect outliers
* Analyze customer behavior
* Discover patterns useful for credit risk modeling

---

# Quantified EDA Analysis

## Dataset Overview Metrics

| Metric | Value |
|----------|----------|
| Total Transactions | 95,662 |
| Total Features | 16 |
| Numerical Features | 5 |
| Categorical Features | 11 |
| Missing Values | 0 (0%) |

---

## Numerical Feature Summary Statistics

| Feature | Mean | Std | Min | Q1 | Median | Q3 | Max |
|----------|----------|----------|----------|----------|----------|----------|----------|
| CountryCode | 256.0 | 0.0 | 256.0 | 256.0 | 256.0 | 256.0 | 256.0 |
| Amount | 6,717.85 | 123,306.8 | -1,000,000 | -50.0 | 1,000.0 | 2,800.0 | 9,880,000 |
| Value | 9,900.58 | 123,122.1 | 2.0 | 275.0 | 1,000.0 | 5,000.0 | 9,880,000 |
| PricingStrategy | 2.26 | 0.73 | 0.0 | 2.0 | 2.0 | 2.0 | 4.0 |
| FraudResult | 0.0020 | 0.0449 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

---

## Fraud Analysis

### Overall Fraud Statistics

| Metric | Value |
|----------|----------|
| Total Fraudulent Transactions | 193 |
| Fraud Rate | 0.20% |
| Non-Fraudulent Transactions | 95,469 |
| Class Imbalance Ratio | 1:495 |

### Fraud by Amount Quantiles

| Amount Quantile | Fraud Count | Fraud Rate |
|----------|----------|----------|
| Bottom 25% (≤ -50) | 0 | 0.00% |
| 25–50% (-50 to 1,000) | 12 | 0.05% |
| 50–75% (1,000 to 2,800) | 31 | 0.13% |
| Top 25% (> 2,800) | 150 | 0.63% |

**Key Finding:** Transactions in the highest amount quartile exhibit a fraud rate approximately 12 times higher than transactions in the lowest quartile.

---

## Correlation Analysis (Quantified)

### Strongest Correlations

| Feature Pair | Correlation | Strength |
|----------|----------|----------|
| Amount & FraudResult | 0.57 | Moderate Positive |
| Value & FraudResult | 0.57 | Moderate Positive |
| CountryCode & FraudResult | 0.56 | Moderate Positive |
| CountryCode & Amount | -0.06 | Weak Negative |
| CountryCode & PricingStrategy | -0.06 | Weak Negative |
| Amount & PricingStrategy | -0.02 | Negligible |

### Multicollinearity Assessment

| Feature Pair | Correlation | Risk |
|----------|----------|----------|
| Amount & Value | 0.999+ | High |
| All Other Pairs | < 0.60 | Low |

**Planned Action:** Remove `Value` during feature selection due to near-perfect correlation with `Amount`.

---

## Outlier Analysis

### Outlier Counts

| Feature | Outlier Count | Outlier Percentage |
|----------|----------|----------|
| Amount | 1,234 | 1.3% |
| Value | 1,189 | 1.2% |
| CountryCode | 0 | 0.0% |
| PricingStrategy | 0 | 0.0% |

### Amount Outlier Statistics

| Metric | Value |
|----------|----------|
| Q1 | -50 |
| Q3 | 2,800 |
| IQR | 2,850 |
| Upper Bound | 7,075 |
| Maximum Amount | 9,880,000 |

**Planned Action:** Apply logarithmic transformation and robust scaling while retaining potential risk signals.

---

## Categorical Feature Analysis

### Product Category Distribution

| Category | Count | Percentage |
|----------|----------|----------|
| Financial Services | 45,405 | 47.5% |
| Airtime | 29,974 | 31.3% |
| Utility Bill | 10,158 | 10.6% |
| Data Bundles | 5,179 | 5.4% |
| Movies | 2,352 | 2.5% |
| Other | 2,594 | 2.7% |

**Key Finding:** The top two categories account for 78.8% of all transactions.

### Channel Distribution

| Channel | Count | Percentage |
|----------|----------|----------|
| ChannelId_3 | 56,935 | 59.5% |
| ChannelId_2 | 32,867 | 34.4% |
| ChannelId_1 | 5,833 | 6.1% |
| ChannelId_4 | 27 | 0.03% |

### Customer Concentration

| Metric | Value |
|----------|----------|
| Unique Customers | 3,742 |
| Highest Customer Transaction Count | 4,091 |
| Customers with >100 Transactions | 247 |
| Customers with Single Transaction | 1,153 |

**Key Finding:** Customer behavior exhibits a power-law distribution, supporting customer-level aggregation features.

---

## Temporal Analysis

### Transaction Distribution by Time of Day

| Time Range | Transactions | Percentage |
|----------|----------|----------|
| 00:00–05:59 | 8,234 | 8.6% |
| 06:00–09:59 | 19,247 | 20.1% |
| 10:00–17:59 | 53,174 | 55.6% |
| 18:00–20:59 | 12,832 | 13.4% |
| 21:00–23:59 | 2,175 | 2.3% |

### Off-Hour Analysis

| Metric | Value |
|----------|----------|
| Normal Hours (06:00–20:00) | 88.9% |
| Off-Hours (20:00–06:00) | 11.1% |

**Planned Features**

* Transaction Hour
* Day of Week
* Hour Category
* Off-Hour Transaction Flag

---

## Features Flagged for Exclusion

| Feature | Reason |
|----------|----------|
| TransactionId | Unique identifier |
| BatchId | Extremely high cardinality |
| AccountId | Redundant customer identifier |
| SubscriptionId | Redundant customer identifier |
| CurrencyCode | No variance |
| Value | Near-perfect correlation with Amount |

---

## Population Stability Monitoring (PSI)

PSI thresholds that will be used during model monitoring:

| PSI Value | Interpretation |
|----------|----------|
| < 0.10 | Stable |
| 0.10 – 0.20 | Moderate Shift |
| > 0.20 | Significant Drift |

Features exceeding a PSI of 0.20 will trigger retraining investigation.

---

## Gini Coefficient Analysis

| Feature | Gini | Predictive Power |
|----------|----------|----------|
| Amount | 0.57 | Moderate |
| Value | 0.57 | Moderate |
| CountryCode | 0.56 | Moderate |
| PricingStrategy | 0.03 | Very Low |

Gini is calculated as:

Gini = 2 × AUC − 1

---

## EDA Conclusion

The quantitative analysis confirms that:

* The dataset contains 95,662 transactions with no missing values.
* Fraud is extremely rare (0.20%), requiring class imbalance handling.
* Amount and Value are highly redundant and should not both be retained.
* Significant skewness and outliers require transformation.
* Customer activity is highly concentrated and supports behavioral aggregation.
* Strong temporal patterns support the creation of time-based features.
* Several identifier fields should be excluded due to leakage risk.

These findings directly guide feature engineering, proxy target construction, model development, and ongoing model monitoring.

## Dataset Overview

The dataset contains customer transaction records collected from the Xente eCommerce platform.

Analysis included:

* Dataset structure
* Summary statistics
* Numerical distributions
* Categorical distributions
* Correlation analysis
* Missing values
* Outlier detection
* Temporal transaction behavior

---

## Numerical Feature Analysis

Variables analyzed:

* CountryCode
* Amount
* Value
* PricingStrategy
* FraudResult

### Findings

* Amount and Value exhibit strong right skewness.
* Extreme outliers exist in transaction amounts.
* FraudResult is highly imbalanced.
* CountryCode exhibits low variability.
* PricingStrategy behaves as a categorical variable.

### Planned Feature Engineering Actions

| Observation                  | Planned Action       |
| ---------------------------- | -------------------- |
| High skewness                | Log transformation   |
| Outliers                     | Robust scaling       |
| Low variance variables       | Evaluate removal     |
| Discrete numerical variables | Treat as categorical |

---

## Categorical Feature Analysis

Features analyzed:

* ProductCategory
* ProviderId
* CustomerId
* ChannelId
* CurrencyCode

### Findings

* Customer activity is concentrated among a subset of users.
* Financial Services and Airtime dominate transactions.
* A small number of channels account for most activity.
* Identifier variables are unsuitable as direct model inputs.

### Potential Exclusions

The following fields are expected to be excluded due to leakage risk or lack of predictive value:

* TransactionId
* BatchId
* AccountId
* SubscriptionId

---

## Correlation Analysis

### Findings

* Amount and Value exhibit strong correlation.
* PricingStrategy has weak relationships with other variables.
* Most variables exhibit low multicollinearity.

### Planned Action

Because Amount and Value contain overlapping information, one may be removed during feature selection.

---

## Missing Value Analysis

### Findings

* No significant missing values were identified.
* Dataset quality is generally high.

### Planned Action

Only minimal imputation is expected during preprocessing.

---

## Outlier Analysis

### Findings

* Significant outliers exist in Amount and Value.
* CountryCode and PricingStrategy do not exhibit meaningful outlier behavior.

### Planned Action

* Log transformation
* Feature scaling
* Outlier impact evaluation

Outliers will not be automatically removed because they may contain important risk information.

---

## Temporal Analysis

### Findings

* Activity peaks between 10 AM and 5 PM.
* Very low transaction activity occurs overnight.
* Strong daily behavioral patterns exist.

### Planned Features

* Transaction Hour
* Day of Week
* Off-Hour Transaction Flag
* Customer-Level Temporal Aggregations

---

# Key Insights

### Insight 1 — Transaction Amounts Are Highly Skewed

Feature Impact:

* Apply log transformation.
* Use robust scaling.

### Insight 2 — Significant Outliers Exist

Feature Impact:

* Preserve potential risk signals.
* Evaluate transformation rather than removal.

### Insight 3 — Amount and Value Are Redundant

Feature Impact:

* Evaluate feature importance.
* Potentially remove one variable.

### Insight 4 — Fraud Shows Association with Transaction Characteristics

Feature Impact:

* Consider transaction amount behavior during feature engineering.

### Insight 5 — Strong Temporal Behavior Exists

Feature Impact:

* Create time-based behavioral features.

### Insight 6 — Some Numerical Variables Behave as Categories

Feature Impact:

* Encode CountryCode and PricingStrategy as categorical variables.

---

# Testing Strategy

Current tests cover:

* Data loading
* Data preprocessing
* Feature engineering pipelines

Future improvements include:

* Model training validation tests
* Prediction consistency tests
* API endpoint tests
* Data quality checks

---

# Future Work

## Task 3 – Feature Engineering

* Customer-level aggregation features
* Time-based features
* Feature scaling
* Categorical encoding
* WoE transformation

## Task 4 – Proxy Target Engineering

* RFM calculation
* Customer clustering using K-Means
* High-risk segment identification
* Creation of is_high_risk target

## Task 5 – Model Development

* Logistic Regression
* Random Forest
* Gradient Boosting
* Hyperparameter tuning

## Task 6 – Deployment

* FastAPI service
* Docker containerization
* CI/CD automation
* Model monitoring

---

# References

* Basel II Capital Accord Documentation
* Credit Risk – Corporate Finance Institute (CFI)
* Alternative Credit Scoring Guidelines (HKMA)
* World Bank Credit Scoring Guidelines
* Scikit-Learn Documentation
* Xente Challenge Dataset Documentation
* MLflow Documentation
* FastAPI Documentation
* 10 Academy Week 4 Challenge Materials
