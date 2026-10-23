"""Streamlit version. Run locally: streamlit run streamlit_app.py
For Streamlit Community Cloud, set main file to streamlit_app.py"""
import joblib
import pandas as pd
import streamlit as st

EMOJI = {"joy": "😄", "sadness": "😢", "anger": "😡",
         "fear": "😨", "love": "❤️", "surprise": "😲"}

@st.cache_resource
def load():
    b = joblib.load("emotion_model.joblib")
    return b["model"], b["labels"]

model, labels = load()
st.set_page_config(page_title="Emotion Detection", page_icon="🎭")
st.title("🎭 Emotion Detection from Text")
st.caption("TF-IDF + Logistic Regression trained on the dair-ai/emotion dataset (6 classes)")

text = st.text_area("Enter a sentence", height=120, placeholder="I am so happy today!")
if st.button("Detect emotion", type="primary") and text.strip():
    probs = model.predict_proba([text])[0]
    top = labels[probs.argmax()]
    st.success(f"Predicted emotion: {EMOJI[top]} **{top.upper()}**  ({probs.max():.1%})")
    st.bar_chart(pd.DataFrame({"probability": probs}, index=labels))
