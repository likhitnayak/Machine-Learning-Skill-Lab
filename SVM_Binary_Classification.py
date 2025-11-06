import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------
# 1. Load and Prepare the Data
# -----------------
# Load the Iris dataset
iris = datasets.load_iris()

# We only want to use the first two features for easy visualization:
# Feature 0: Sepal Length
# Feature 1: Sepal Width
X = iris.data[:, :2] 
y = iris.target

# We also want to create a binary classification problem by only using two classes:
# Class 0: setosa
# Class 1: versicolor
# We will filter the data to keep only these two classes.
X = X[y != 2]
y = y[y != 2]

# -----------------
# 2. Split Data into Training and Testing Sets
# -----------------
# This step divides our data into a set for training the model and a separate set for testing it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------
# 3. Train a Linear SVM Model
# -----------------
# Create an instance of the SVM classifier with a linear kernel.
# The 'C' parameter is the regularization parameter. For now, we'll leave it at the default of 1.0.
linear_svm = svm.SVC(kernel='linear', C=1.0)

# Train the model on the training data
print("Training the linear SVM model...")
linear_svm.fit(X_train, y_train)
print("Model training complete.")

# -----------------
# 4. Visualize the Decision Boundary
# -----------------
print("Visualizing the decision boundary...")

# Create a mesh grid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict the class for each point in the mesh grid
Z = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the training data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM with Linear Kernel Decision Boundary')

# Highlight the support vectors
plt.scatter(linear_svm.support_vectors_[:, 0], linear_svm.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', linewidth=1.5)

plt.show()

# -----------------
# 5. Evaluate the Model
# -----------------
# Make predictions on the test data
y_pred = linear_svm.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")