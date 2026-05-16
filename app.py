import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Load Dataset from URL
# -----------------------------
url = "https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv"

df = pd.read_csv(url)

# -----------------------------
# Features and Labels
# -----------------------------
X = df["text"]
y = df["labels"]

# -----------------------------
# Text Vectorization
# -----------------------------
cv = CountVectorizer()

X_transformed = cv.fit_transform(X)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_transformed, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Article Classification")

st.write("Enter a news article or headline below.")

# -----------------------------
# Input Box
# -----------------------------
user_input = st.text_area(
    "Enter News Text",
    height=200
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Category"):

    if user_input.strip() != "":

        transformed_input = cv.transform([user_input])

        prediction = model.predict(transformed_input)

        st.success(f"Predicted Category: {prediction[0]}")

    else:
        st.warning("Please enter some text.")