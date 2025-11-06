import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------
# 1. Load and Prepare the Data
# -----------------
# Load the Iris dataset
iris = datasets.load_iris()

# For this exercise, we will use petal length and petal width for visualization
# Feature 2: Petal Length
# Feature 3: Petal Width
X = iris.data[:, 2:] 
y = iris.target

# -----------------
# 2. Split Data into Training and Testing Sets
# -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------
# 3. Train a Linear SVM Model for Multi-Class Classification
# -----------------
# Scikit-learn's SVC handles multi-class classification using a one-vs-one scheme by default.
multi_class_svm = svm.SVC(kernel='linear', C=1.0)

# Train the model
print("Training the multi-class linear SVM model...")
multi_class_svm.fit(X_train, y_train)
print("Model training complete.")

# -----------------
# 4. Visualize the Decision Boundaries
# -----------------
print("Visualizing the decision boundaries...")

# Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh grid
Z = multi_class_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, cmap=plt.cm.jet, alpha=0.3)

# Plot the training data points
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.jet, edgecolors='k')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Multi-Class SVM with Linear Kernel Decision Boundaries')

# Create a legend
handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names)

plt.show()

# -----------------
# 5. Evaluate the Model
# -----------------
# Make predictions on the test data
y_pred = multi_class_svm.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# Generate and display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)