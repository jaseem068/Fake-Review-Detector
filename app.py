from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"error": "Please enter a review"})

    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]

    if str(prediction).lower() in ["fake", "1", "fake review"]:
        result = "Fake Review"
    else:
        result = "Genuine Review"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)