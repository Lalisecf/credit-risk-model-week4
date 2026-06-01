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