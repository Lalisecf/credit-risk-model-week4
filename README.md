# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end Credit Risk Scoring system for Bati Bank using transaction data from an eCommerce platform.

The objective is to identify high-risk customers, estimate default probability, generate credit scores, and support buy-now-pay-later lending decisions.

---

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
├── notebooks/
├── src/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
```
### Credit Scoring Business Understanding
1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to maintain transparent, reliable, and auditable credit risk assessment processes. Credit risk models directly influence lending decisions, regulatory capital requirements, and overall risk management practices.

Because these decisions can affect customers and financial stability, regulators expect institutions to understand how predictions are generated. Therefore, credit scoring models must be interpretable, well documented, and reproducible.

Interpretability enables risk managers to explain why a customer was classified as high-risk or low-risk. Documentation provides evidence of model assumptions, feature selection, validation procedures, performance monitoring, and limitations. These practices ensure regulatory compliance, facilitate audits, and support ongoing model governance.

2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The provided transaction dataset does not contain historical loan repayment outcomes or default indicators. Since supervised machine learning requires a target variable, a proxy variable must be created to approximate customer credit risk.

A practical approach is to use customer behavioral patterns derived from Recency, Frequency, and Monetary (RFM) metrics. Customers with low engagement, infrequent transactions, and low transaction value may be considered higher risk relative to more active customers.

However, proxy targets introduce several risks:

The proxy may not accurately represent actual loan default behavior.
Misclassification can occur if inactive customers are financially healthy.
The model may learn behavioral patterns instead of true repayment risk.
Lending decisions based on an imperfect proxy could increase financial losses or unfairly reject qualified applicants.

Therefore, proxy-based predictions should be treated as assumptions rather than ground truth and continuously validated when real repayment data becomes available.

3. What are the key trade-offs between a simple interpretable model and a high-performance model in a regulated financial context?
Logistic Regression with WoE

Advantages:

Highly interpretable
Easy to explain to regulators
Stable and transparent
Supports scorecard development
Easier to document and monitor

Disadvantages:

May fail to capture complex nonlinear relationships
Often produces lower predictive performance
Gradient Boosting Models

Advantages:

Higher predictive accuracy
Captures nonlinear interactions
Handles complex customer behavior patterns

Disadvantages:

Less transparent
More difficult to explain individual predictions
Harder to validate and document
Greater regulatory scrutiny
Recommended Approach

In regulated financial environments, model selection requires balancing predictive performance and interpretability. Logistic Regression combined with Weight of Evidence (WoE) transformations is often preferred because it provides transparency and regulatory acceptance.

However, Gradient Boosting models can be evaluated alongside Logistic Regression to determine whether performance gains justify the additional complexity. Explainability tools such as SHAP can help improve transparency when using advanced machine learning models.
## Exploratory Data Analysis (Task 2)

### Objective

The objective of the exploratory data analysis (EDA) phase was to understand the structure of the dataset, identify data quality issues, discover behavioral patterns, detect outliers, and generate insights that would guide feature engineering and model development.

---

### Dataset Overview

The Xente transaction dataset contains customer transaction records collected from an eCommerce platform. The dataset includes transaction identifiers, customer information, product details, transaction amounts, timestamps, pricing strategies, and fraud indicators.

The analysis focused on:

* Data structure and data types
* Summary statistics
* Numerical feature distributions
* Categorical feature distributions
* Correlation analysis
* Missing value assessment
* Outlier detection
* Temporal transaction patterns

---

### Numerical Feature Analysis

The numerical variables examined were:

* CountryCode
* Amount
* Value
* PricingStrategy
* FraudResult

#### Findings

* **Amount** and **Value** exhibit strong positive skewness with several extreme outliers.
* Most transactions involve relatively small amounts, while a small number of transactions account for very large values.
* **FraudResult** is highly imbalanced, with non-fraudulent transactions significantly outnumbering fraudulent transactions.
* **CountryCode** shows very little variation, indicating limited predictive information.
* **PricingStrategy** contains a small number of discrete values and behaves more like a categorical feature than a continuous numerical variable.

---

### Categorical Feature Analysis

The categorical variables examined included:

* TransactionId
* BatchId
* AccountId
* SubscriptionId
* CustomerId
* CurrencyCode
* ProviderId
* ProductId
* ProductCategory
* ChannelId
* TransactionStartTime

#### Findings

* Customer activity is concentrated among a relatively small subset of customers.
* Financial Services and Airtime dominate transaction activity within ProductCategory.
* A small number of channels account for the majority of transactions.
* CurrencyCode contains only one dominant value, suggesting limited usefulness for prediction.
* Identifier fields such as TransactionId and BatchId serve primarily as unique identifiers and are not suitable as direct model features.

---

### Correlation Analysis

Correlation analysis was performed on numerical variables to identify linear relationships between features.

#### Findings

* Amount shows the strongest relationship with FraudResult among the available numerical variables.
* PricingStrategy exhibits very weak correlations with other variables.
* Most feature pairs have low correlation values, indicating limited multicollinearity.
* Low multicollinearity supports the use of multiple engineered features during model development.

---

### Outlier Analysis

Boxplots were used to identify unusual observations and extreme values.

#### Findings

* Significant outliers were observed in Amount and Value.
* Transaction values span a wide range, suggesting the need for transformation and scaling.
* CountryCode and PricingStrategy contain discrete values and do not exhibit traditional outlier behavior.
* FraudResult is a binary variable, making standard outlier analysis inapplicable.

#### Planned Actions

* Apply log transformation to highly skewed features.
* Standardize or normalize numerical variables.
* Evaluate redundancy between Amount and Value during feature selection.

---

### Temporal Pattern Analysis

Transaction timestamps were analyzed by extracting transaction hours.

#### Findings

* Transaction activity is highest during standard business hours (10:00 AM–5:00 PM).
* Activity gradually increases during the morning and decreases during the evening.
* Very few transactions occur overnight.
* Transaction timing may contain valuable behavioral information for customer risk profiling.

#### Planned Actions

Future feature engineering will include:

* Transaction hour
* Off-hour transaction indicators
* Customer-level transaction timing statistics

---

### Key Insights

#### Insight 1: Transaction Amounts Are Highly Skewed

Amount and Value contain significant skewness and extreme outliers. Appropriate transformations such as log scaling will be required before model training.

#### Insight 2: Customer Activity Is Concentrated in Specific Categories

A small number of product categories and transaction channels account for most platform activity, indicating potentially valuable behavioral signals.

#### Insight 3: Fraudulent Transactions Are Rare

FraudResult is highly imbalanced, reinforcing the need to create a separate proxy target variable for credit risk prediction.

#### Insight 4: Low Multicollinearity Exists Among Features

Most numerical features show weak pairwise correlations, which is beneficial for model stability and interpretability.

#### Insight 5: Customer Behavior Follows Strong Daily Patterns

Transaction activity is concentrated during business hours, suggesting that time-based behavioral features may improve customer risk segmentation.

#### Insight 6: Some Numerical Variables Behave as Categories

CountryCode and PricingStrategy contain discrete values and should be treated as categorical variables during preprocessing.

#### Insight 7: Amount and Value May Be Redundant

Because Value represents the absolute value of Amount, both variables may contain overlapping information and should be evaluated during feature selection.

---

### Impact on Feature Engineering

The EDA findings directly inform the next stage of the project:

* Create customer-level aggregate features.
* Extract temporal features from transaction timestamps.
* Encode categorical variables using appropriate techniques.
* Transform and scale skewed numerical variables.
* Engineer RFM (Recency, Frequency, Monetary) features for proxy target creation.
* Evaluate redundant and low-variance features before model training.

These findings establish the foundation for Task 3 (Feature Engineering) and Task 4 (Proxy Target Variable Engineering).
