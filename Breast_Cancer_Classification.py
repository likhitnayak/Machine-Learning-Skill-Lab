import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load the breast cancer dataset
cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model.
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]
print("True Negatives (TN):", TN, "- Correctly predicted as not having cancer.")
print("False Positives (FP):", FP, "- Incorrectly predicted as having cancer.")
print("False Negatives (FN):", FN, "- Incorrectly predicted as not having cancer.")
print("True Positives (TP):", TP, "- Correctly predicted as having cancer.")
print("-" * 50)

accuracy = (TN + TP) / (TN + FP + FN + TP)
print("Accuracy:", accuracy)
sensitivity = TP / (TP + FN)
print("Sensitivity:", sensitivity)
specificity = TN / (TN + FP)
print("Specificity:", specificity)
fpr = FP / (FP + TN)
print("False Positive Rate:", fpr)
f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)
print("F1 Score:", f1_score)

# Plot ROC curve and calculate AUC
postiive_probabilities = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, postiive_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
# plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

roc_auc = auc(fpr, tpr)
print(f"Area Under the Curve (AUC): {roc_auc:.4f}")