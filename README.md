# Sentiment Analysis from Scratch

## Project Description

This project implements a complete Sentiment Analysis system from scratch without relying on pre-trained models or APIs. It uses traditional NLP techniques and a Naive Bayes classifier to determine whether text expresses positive, negative, or neutral sentiment.

The goal is to understand the fundamentals of text classification and natural language processing.

## What You'll Learn

1. Text preprocessing (tokenization, stopword removal, stemming)
2. Building a vocabulary from scratch
3. Feature extraction using Bag of Words (BoW) and TF-IDF
4. Training a Naive Bayes classifier
5. Building a complete prediction pipeline
6. Evaluating model performance

## Technologies Used

- Python 3.x
- NumPy
- Pandas
- NLTK (for text processing)
- Scikit-learn (for machine learning)
- Matplotlib (for visualization)

## How to Run

```bash
cd sentiment-analysis
pip install -r requirements.txt
python sentiment_analysis.py
```

Or use the Jupyter notebook:

```bash
jupyter notebook sentiment_analysis.ipynb
```

## Project Structure

- `sentiment_analysis.py` - Main Python script
- `sentiment_analysis.ipynb` - Jupyter notebook version
- `data/` - Dataset folder
- `models/` - Trained model and vectorizer

## How It Works

1. **Data Collection**: Using movie reviews dataset (positive/negative reviews)
2. **Text Preprocessing**:
   - Lowercasing
   - Removing punctuation
   - Tokenization
   - Stopword removal
   - Stemming

3. **Feature Extraction**:
   - Build vocabulary from training data
   - Convert text to numerical features using BoW or TF-IDF

4. **Model Training**:
   - Train Naive Bayes classifier
   - Evaluate on test data

5. **Prediction**:
   - Preprocess new text
   - Transform to features
   - Predict sentiment

## Example Usage

```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.train('data/reviews.csv')
result = analyzer.predict("This movie was absolutely amazing!")
print(result)  # Output: positive
```

## Techniques Used

- **Preprocessing**: NLTK's word_tokenize, PorterStemmer
- **Vectorization**: CountVectorizer / TfidfVectorizer
- **Classification**: Multinomial Naive Bayes
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

## License

MIT License