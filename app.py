import streamlit as st
import pandas as pd
from sentiment_analysis import SentimentAnalyzer

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Sentiment Analysis from Scratch")
st.write("Analyze text sentiment using Naive Bayes classifier")

analyzer = SentimentAnalyzer()
analyzer.train()

st.sidebar.header("About")
st.sidebar.write("""
- Text Preprocessing
- TF-IDF Vectorization  
- Naive Bayes Classifier
- Built from Scratch!
""")

st.subheader("Enter Text to Analyze")
text_input = st.text_area("Type your text here:", height=100)

if st.button("Analyze Sentiment"):
    if text_input:
        result = analyzer.predict(text_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Result")
            if result['sentiment'] == 'positive':
                st.success("✅ POSITIVE")
            else:
                st.error("❌ NEGATIVE")
        
        with col2:
            st.subheader("Confidence")
            st.write(f"{result['confidence']*100:.1f}%")
        
        st.subheader("Probabilities")
        st.bar_chart(result['probabilities'])

st.subheader("Sample Predictions")
samples = [
    "I absolutely loved this movie great acting",
    "Terrible film waste of time",
    "Good movie with excellent storyline",
    "Very boring and bad experience",
]

for sample in samples:
    result = analyzer.predict(sample)
    emoji = "✅" if result['sentiment'] == 'positive' else "❌"
    st.write(f"{emoji} {sample}")