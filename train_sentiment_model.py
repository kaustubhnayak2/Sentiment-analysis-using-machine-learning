import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


# Load IMDb dataset
def load_imdb_dataset(base_dir):
    data = {"review": [], "sentiment": []}
    for label in ["pos", "neg"]:
        path = os.path.join(base_dir, label)
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding="utf-8") as f:
                data["review"].append(f.read())
                data["sentiment"].append(label)
    return pd.DataFrame(data)

# Load train data
train_dir = "aclImdb/train"
df = load_imdb_dataset(train_dir)

# Preprocessing
df['sentiment'] = df['sentiment'].map({'pos': 1, 'neg': 0})
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression())
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "sentiment_model.pkl")
print("Model saved as sentiment_model.pkl")
