import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load real dataset
df = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'text'])

# Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Test
predictions = model.predict(X_test_vectors)

# Accuracy and report
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nDetailed Report:\n", classification_report(y_test, predictions))

# Try your own message
sample = ["Congratulations! You’ve been selected for a FREE MacBook Pro. Click here to claim your prize"]
sample_vector = vectorizer.transform(sample)
print("\nSample Prediction:", model.predict(sample_vector))  # 1 = spam
