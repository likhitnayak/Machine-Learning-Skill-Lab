import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the 20 Newsgroups dataset.
# The 20 newsgroups dataset comprises around 18000 news posts on 20 different topics split in two subsets: one for training and the other one for testing.
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

# Convert text data into numerical features using CountVectorizer
# CountVectorizer is a tool from the scikit-learn library used to transform a collection of text documents into a numerical representation.
# Machine learning models work with numbers, not text, so this conversion is a crucial first step in text analysis.
# How it works:
# 1. Tokenization: It breaks down the text of each document into individual words, which are called tokens. It also performs basic preprocessing like converting all words to lowercase and removing punctuation.
# 2. Vocabulary Building: It builds a vocabulary of all the unique words found in the entire collection of training documents. Each unique word is assigned a specific index.
# 3. Counting: For each document, it counts the occurrences of each word from the vocabulary.
# 4. Vector Creation: It creates a numerical vector for each document. The length of this vector is the total number of unique words in the vocabulary. Each element in the vector corresponds to a word in the vocabulary, and its value is the count of how many times that word appeared in the document.

# The result is a matrix where rows represent documents and columns represent unique words (the vocabulary), and the cell values are the word counts. This is often called a "document-term matrix".
vectorizer = CountVectorizer()

# The fit_transform method first learns the vocabulary from the training data (fit)
# and then creates the document-term matrix for the training data (transform).
X_train = vectorizer.fit_transform(newsgroups_train.data)

# For the test data, we only use the transform method.
# This ensures that the test data is vectorized using the same vocabulary learned from the training data. Any words in the test data that were not in the training data are ignored.
X_test = vectorizer.transform(newsgroups_test.data)

# Fit a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, newsgroups_train.target)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(newsgroups_test.target, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate the confusion matrix
cm = confusion_matrix(newsgroups_test.target, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=newsgroups_train.target_names, yticklabels=newsgroups_train.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()