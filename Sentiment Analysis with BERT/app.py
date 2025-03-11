import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

st.title("Sentiment Analysis with BERT")
st.write("Enter text below and let BERT determine its sentiment!")

user_input = st.text_area("Enter text here:", "")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        output = model(**inputs)
    
    scores = F.softmax(output.logits, dim=1).squeeze().tolist()

    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    
    st.write("### Sentiment Scores:")
    for label, score in zip(labels, scores):
        st.write(f"**{label}:** {score:.2%}")

    predicted_sentiment = labels[scores.index(max(scores))]
    st.success(f"Predicted Sentiment: **{predicted_sentiment}** ")
