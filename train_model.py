import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple training data
data = {
    "review": [
        "This product is excellent and works very well",
        "Good quality product and useful",
        "I have used this product for two weeks and I am satisfied",
        "The product arrived on time and works properly",
        "Very happy with my purchase",
        "The quality is good for the price",
        "Amazing product, I really liked it",
        "The product stopped working after one day",
        "Worst product ever, do not buy this",
        "Best product in the world buy now",
        "Amazing amazing amazing five stars buy immediately",
        "This is the best product ever made",
        "Excellent excellent excellent product",
        "I received a damaged product",
        "The product quality is poor",
        "It did not work as expected"
    ],
    "label": [
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "genuine",
        "fake",
        "fake",
        "fake",
        "fake",
        "genuine",
        "genuine",
        "genuine"
    ]
}

df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["review"])
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")
print("fake_review_model.pkl created!")
print("vectorizer.pkl created!")