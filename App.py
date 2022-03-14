from flask import Flask, request, render_template
import pandas as pd
import joblib



# load your model
model = joblib.load('Aim-Technology/decisionTree.pkl')
vectorizer = joblib.load('Aim-Technology/TFIDF.pkl')
label_encoder = joblib.load('Aim-Technology/labelIncoder.pkl')


# Initiate an object
app = Flask(__name__)


# app pages
# home page
@app.route('/')
def welcome_page():
    return render_template("Home.html")


# prediction page
@app.route('/prepare')
def render_predict_page():
    return render_template("input_feature.html")


# prediction
@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        features = request.form["features"]
        # Convert text into series
        test_text_series = pd.Series(features, index=[0])

        # TFIDF transformation
        transformed_test_text = vectorizer.transform(test_text_series)

        # Predict
        predicted_label = model.predict(transformed_test_text)

        # Print the predicted label
        prediction = label_encoder.classes_[predicted_label]
    return render_template("input_feature.html", prediction=prediction[0])


if __name__ == '__main__':
    app.run(port=9999,
            debug=True
            )


# sample = وينك شو الاخبار
# sample = وانت شو تسوي داحين؟


