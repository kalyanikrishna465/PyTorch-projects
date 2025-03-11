import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

MODEL_NAME = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

questions = [
    "I enjoy trying new things and exploring new ideas.",
    "I am always prepared and pay attention to details.",
    "I feel energized when around people and social gatherings.",
    "I try to be helpful and sensitive to others' needs.",
    "I often feel stressed or anxious in daily situations."
]

st.title("Interactive Psychometric Analysis")
st.write("Answer the following questions to analyze your personality.")

responses = []
for i, question in enumerate(questions):
    response = st.text_area(f"**{question}**", key=f"q{i}")
    responses.append(response)

if st.button("Analyze Personality"):
    if any(response.strip() == "" for response in responses):
        st.warning("Please answer all questions before proceeding.")
    else:
        user_input = " ".join(responses)
        
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            output = model(**inputs)

        scores = F.softmax(output.logits, dim=1).squeeze().tolist()

        st.write("### Personality Trait Scores:")
        for trait, score in zip(traits, scores):
            st.write(f"**{trait}:** {score:.2%}")

        dominant_trait = traits[scores.index(max(scores))]
        st.success(f"Your Dominant Trait: **{dominant_trait}** ")
