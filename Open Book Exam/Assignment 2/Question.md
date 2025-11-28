## Problem Statement

You are a data scientist at "FinSecure," a peer-to-peer lending company that connects borrowers with investors. The company's success hinges on its ability to accurately assess the risk of a borrower defaulting on their loan. A high rate of loan defaults not only results in financial losses for investors but also damages the company's reputation and trust in the marketplace.

The loan approval team currently relies on a manual, scorecard-based system to approve or deny loan applications. However, this process is slow and may not be capturing complex, non-linear patterns in the data that could indicate a higher risk of default. The leadership team wants to develop a data-driven system to predict the likelihood of a loan being paid back in full.

Your task is to analyze the provided loan application dataset and build a predictive model to identify loans that are likely to default. You are free to choose any architecture you have learned in class.


## The Dataset

You are provided with a dataset named loan_data.csv. It contains historical data on loan applications, including applicant details and whether the loan was ultimately paid back.

- id: A unique identifier for the loan application.
- annual_income: The annual income of the borrower.
- debt_to_income_ratio: The ratio of the borrower's total debt to their annual income.
- credit_score: The credit score of the borrower.
- loan_amount: The total amount of the loan requested.
- interest_rate: The interest rate assigned to the loan.
- gender: The gender of the borrower.
- marital_status: The marital status of the borrower.
- education_level: The highest level of education attained by the borrower.
- employment_status: The employment status of the borrower.
- loan_purpose: The stated purpose of the loan (e.g., debt consolidation, home improvement).
- grade, subgrade: A loan grade and subgrade assigned by FinSecure, indicating risk.
- loan_paid_back: The target variable; indicates whether the loan was paid back (1) or not (0).

## Your Deliverables

**Section 1: Problem Formulation & Target Variable Analysis**

The business problem is to predict loan default. The dataset provides a target variable "loan_paid_back". Your first task is to understand what a "default" means in this context.

**Section 2: Feature Engineering & Preprocessing Pipeline**

Based on the raw data, you have to design a comprehensive preprocessing pipeline that will handle the following:
- Missing values (if any are present).
- Categorical feature encoding for columns like "loan_purpose" and "education_level" using your choice of encoder (e.g., One-Hot Encoding, Ordinal Encoding).
- Feature scaling / normalization for numerical columns like "annual_income" and "loan_amount"

Also, keep in mind that you have split your data into training set and testing set.

**Section 3: Model Development and Tuning**

For each sample (id) in your test set, your model must predict a probability for the "loan_paid_back" variable. The goal should be to maximize the area under the ROC curve (AUC) between the predicted probability and the actual value.

**Section 4: Subgroup Analysis**

The loan approval team needs to ensure the model is fair and doesn't disproportionately penalize certain groups. Calculate and present the following metrics for your final model (using your chosen threshold) across these subgroups:
- By education_level: (Report on each level present in the data)
- By loan_purpose: (Report on the top 3 and bottom 3 performing purposes)

Metrics to report: AUC
