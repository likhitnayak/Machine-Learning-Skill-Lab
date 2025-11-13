## Problem Statement

You work for a Turkish e-commerce marketplace. Your leadership wants a churn prediction that flags customers at risk of not purchasing again within a future horizon, so retention campaigns can be targeted. As you know, customer retention is critical to the long-term success of any marketplace. The company is currently facing a challenge where customers are becoming inactive ("churning") without any warning. Your marketing team wants to launch a targeted retention campaign (e.g., offering special discounts, personalized outreach), and to do that, they need a reliable, data-driven system to predict which of the active customers are at the highest risk of churning in the near future.

A crucial requirement from the leadership is explainability. The campaign managers will not use a "black box" system. They need to understand the key factors driving the predictions to build trust and potentially tailor their campaign messaging. For this reason, you have a strict technical constraint - you must develop a solution using a single supervised model architecture. You are free to choose the architecture (e.g., Logistic Regression, SVM, Random Forest, etc.), but you are not permitted to blend, stack, or ensemble different model architectures. However, orchestrating multiple models of the same family or using rule-based systems is permissible if justified. 

Your task is to analyze the provided transaction dataset and build a predictive model for churn.

## The Dataset

You are provided with a dataset named marketplace_transactions.csv. It contains 18 columns detailing customer orders and behavior from 2023-01-01 to 2024-03-26, where each customer has multiple orders

- Order Information: Order_ID, Date
- Customer Demographics: Customer_ID, Age, Gender, City
- Product Information: Product_Category, Unit_Price, Quantity
- Transaction Details: Discount_Amount, Total_Amount, Payment_Method
- Customer Behavior Metrics: Device_Type, Session_Duration_Minutes, Pages_Viewed, Is_Returning_Customer
- Post-Purchase Metrics: Delivery_Time_Days, Customer_Rating

## Your Deliverables

**Section 1: Problem Formulation & Churn Definition**

You have to define a metric for churn and provide a strong justification for your choice. While doing this, consider business cycles, product purchase frequency, and data availability.
You need to decide on the unit of analysis for your model. Will you create one snapshot per customer (e.g., based on their last transaction before a specific cutoff date)? Or will you create periodic snapshots (e.g., one snapshot per active customer per month)? Define your rule and justify why it's appropriate for this business problem.

Based on your definitions above, construct the target variable (churn = 1 or 0) for each snapshot. Crucially, explain your methodology for selecting a cutoff date and ensuring you are not peeking into the future when creating these labels (no data leakage).

**Section 2: Feature Engineering & Preprocessing Pipeline**

The raw data is just a starting point and you are allowed to engineer new, meaningful features. Think about RFM (Recency, Frequency, Monetary) metrics, behavioral patterns, or aggregations over a customer's history. Examples: days_since_last_order, avg_time_between_orders, total_spend, preferred_product_category, etc.

Once you have decided the set of features (with justifications), you have to build a pipeline that takes care of all the preprocessing workflow. This pipeline should handle:
- Missing values (if any).
- Categorical feature encoding (e.g., One-Hot Encoding, Target Encoding).
- Numerical feature scaling (e.g., StandardScaler, MinMaxScaler).

**Section 3: Model Development and Tuning**
1) Model Selection: Choose two distinct model families that are well-suited for this classification task and adhere to the "single architecture" constraint (e.g., Family 1: Logistic Regression, Family 2: SVM).
2) Model Training & Hyperparameter Tuning: For each of the two model families, train the model on your prepared dataset. Implement a hyperparameter tuning strategy (e.g., GridSearchCV, RandomizedSearchCV) to find the optimal set of parameters. The primary optimization metric must be the Area Under the Precision-Recall Curve (PR-AUC). Justify why PR-AUC is a more suitable metric than ROC-AUC for this business problem 
3) Model Comparison: Present the best PR-AUC scores for each of the two tuned models. Select the single best-performing model that you will recommend to the stakeholders.

**Section 4: Evaluation, Thresholding, and Subgroup Analysis**

The default prediction threshold of 0.5 is rarely optimal for a business problem. Plot the Precision-Recall curve for your final model and choose a specific decision threshold (e.g., 0.35). Justify your choice from a business perspective. (e.g., "We chose a threshold of 0.4 because it allows us to identify 60% of true churners (Recall) while ensuring that 75% of our flagged customers are indeed at risk (Precision), which is an acceptable trade-off for our marketing budget.")

Since the stakeholders are concerned about model fairness and want to ensure the model works equally well across different customer segments, you have to calculate and present the following metrics for your final model (using your chosen threshold) for these subgroups:
By Device_Type: (Mobile, Desktop, Tablet)
By City: (Report on the top 3 and bottom 3 performing cities)
Metrics to report: PR-AUC, Precision, Recall, and F1-Score.
