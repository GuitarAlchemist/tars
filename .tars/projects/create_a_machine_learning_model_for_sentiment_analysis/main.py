Here is the complete, functional code for `main.py`:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Load configuration file
with open('config.json') as f:
    config = json.load(f)

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_data['text'], train_data['sentiment'], test_size=0.2, random_state=config['random_state'])

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluate model on testing set
y_pred = clf.predict(vectorizer.transform(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model to file
joblib.dump(clf, 'models/model.pkl')

print("Model saved successfully!")
```

This code assumes that the `config.json` file contains the following configuration:

```json
{
    "random_state": 42,
    ...
}
```

You can modify this code as needed to fit your specific requirements.