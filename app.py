import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("🎭 Sentiment Analysis App")
st.subheader("Enter a review to analyze its sentiment")

# Text input
user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():  # Check if input is not empty
        # Transform input text using the same vectorizer
        text_vectorized = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(text_vectorized)[0]
        
        # Display result
        if prediction == 1:
            st.success("✅ Sentiment: Positive")
        else:
            st.error("❌ Sentiment: Negative")
    else:
        st.warning("⚠️ Please enter some text!")
