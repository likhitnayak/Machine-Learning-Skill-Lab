from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Fit a DecisionTreeClassifier and select the best criterion between gini and entropy.
criteria = ['gini', 'entropy']
for criterion in criteria:
    classifier = DecisionTreeClassifier(criterion=criterion, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Criterion: {criterion}, Accuracy: {accuracy:.4f}")
    # Generate and display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # 4. Plot the Final Tree
    plt.figure(figsize=(10, 6))
    plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, fontsize=10)
    plt.title(f'Decision Tree for Iris Classification (Criterion: {criterion})')
    plt.show()


