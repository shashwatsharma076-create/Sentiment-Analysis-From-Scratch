import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TextPreprocessor:
    def __init__(self):
        self.stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
            "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
            "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
            "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
            "won't", 'wouldn', "wouldn't"
        ])
        
        self.stemming_rules = {}
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stopwords]
    
    def simple_stem(self, word):
        if word in self.stemming_rules:
            return self.stemming_rules[word]
        
        suffixes = ['ing', 'ed', 's', 'es', 'ly', 'ment', 'tion', 'ness', 'ful', 'able']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stemmed = word[:-len(suffix)]
                if len(stemmed) >= 3:
                    self.stemming_rules[word] = stemmed
                    return stemmed
        
        self.stemming_rules[word] = word
        return word
    
    def preprocess(self, text):
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = [self.simple_stem(token) for token in tokens]
        return ' '.join(tokens)

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.vocab = None
    
    def create_sample_data(self):
        positive_reviews = [
            "This movie is absolutely amazing I loved every moment of it",
            "Great film with fantastic acting and compelling story",
            "One of the best movies I've ever seen highly recommend",
            "Wonderful experience the director did an excellent job",
            "Brilliant performance by the cast truly captivating",
            "Outstanding movie with great music and cinematography",
            "This film exceeded my expectations loved it",
            "Fantastic story very entertaining and moving",
            "Excellent movie with deep meaning and great visuals",
            "Absolutely loved this film best one this year",
        ]
        
        negative_reviews = [
            "This movie is terrible waste of time and money",
            "Boring film with bad acting and poor storyline",
            "One of the worst movies I have ever seen",
            "Terrible acting the script was horrible",
            "Very disappointing movie not worth watching",
            "Awful film complete waste of time",
            "Bad movie with no plot and terrible direction",
            "Horrible experience I regret watching this",
            "Very poor quality acting and cinematography",
            "Disappointing film nothing good about it",
        ]
        
        data = []
        for review in positive_reviews:
            data.append({'text': review, 'sentiment': 'positive'})
        for review in negative_reviews:
            data.append({'text': review, 'sentiment': 'negative'})
        
        np.random.seed(42)
        for _ in range(100):
            pos_template = np.random.choice([
                "Great {noun} {verb} {adj}",
                "Absolutely {adj} {noun}",
                "Best {noun} ever",
            ])
            neg_template = np.random.choice([
                "Terrible {noun} {verb}",
                "Very {adj} {noun}",
                "Awful {noun}",
            ])
            
            nouns = ["movie", "film", "scene", "acting", "story", "plot", "director"]
            verbs = ["is", "was", "felt"]
            adjs = ["amazing", "terrible", "boring", "good", "bad", "great", "horrible"]
            
            pos_sentiment = 'positive'
            neg_sentiment = 'negative'
            
            if np.random.random() > 0.5:
                data.append({
                    'text': f"I think this is a {np.random.choice(adjs)} {np.random.choice(nouns)}",
                    'sentiment': pos_sentiment
                })
            else:
                data.append({
                    'text': f"I think this is a {np.random.choice(adjs)} {np.random.choice(nouns)}",
                    'sentiment': neg_sentiment
                })
        
        df = pd.DataFrame(data)
        
        os.makedirs('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/sentiment-analysis/data', exist_ok=True)
        df.to_csv('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/sentiment-analysis/data/reviews.csv', index=False)
        
        return df
    
    def train(self, data_path=None):
        print("\n" + "=" * 60)
        print("TRAINING SENTIMENT ANALYSIS MODEL")
        print("=" * 60)
        
        if data_path is None:
            df = self.create_sample_data()
        else:
            df = pd.read_csv(data_path)
        
        print(f"\nLoaded {len(df)} reviews")
        print(f"Positive: {len(df[df['sentiment'] == 'positive'])}")
        print(f"Negative: {len(df[df['sentiment'] == 'negative'])}")
        
        print("\nPreprocessing text...")
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        print("Sample preprocessed texts:")
        for i in range(3):
            print(f"  Original: {df['text'].iloc[i]}")
            print(f"  Processed: {df['processed_text'].iloc[i]}")
            print()
        
        X = df['processed_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\nBuilding TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(max_features=500)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        print("\nTraining Naive Bayes classifier...")
        self.model = MultinomialNB()
        self.model.fit(X_train_vec, y_train)
        
        y_pred = self.model.predict(X_test_vec)
        
        print("\n" + "-" * 40)
        print("MODEL EVALUATION")
        print("-" * 40)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        
        return accuracy
    
    def predict(self, text):
        if self.model is None:
            print("Model not trained. Training first...")
            self.train()
        
        processed = self.preprocessor.preprocess(text)
        vec = self.vectorizer.transform([processed])
        prediction = self.model.predict(vec)[0]
        probabilities = self.model.predict_proba(vec)[0]
        
        return {
            'text': text,
            'sentiment': prediction,
            'confidence': max(probabilities),
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def predict_batch(self, texts):
        if self.model is None:
            print("Model not trained. Training first...")
            self.train()
        
        processed = [self.preprocessor.preprocess(text) for text in texts]
        vec = self.vectorizer.transform(processed)
        predictions = self.model.predict(vec)
        probabilities = self.model.predict_proba(vec)
        
        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            results.append({
                'text': text,
                'sentiment': pred,
                'confidence': max(probabilities[0]),
                'probabilities': dict(zip(self.model.classes_, probs))
            })
        
        return results

def visualize_results(analyzer, test_texts):
    sentiments = []
    for text in test_texts:
        result = analyzer.predict(text)
        sentiments.append(result['sentiment'])
    
    labels = ['Positive', 'Negative']
    counts = [sentiments.count('positive'), sentiments.count('negative')]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(labels, counts, color=['green', 'red'], edgecolor='black')
    axes[0].set_title('Sentiment Distribution')
    axes[0].set_ylabel('Count')
    
    colors = ['green' if s == 'positive' else 'red' for s in sentiments]
    axes[1].bar(range(len(test_texts)), [1]*len(test_texts), color=colors)
    axes[1].set_title('Prediction per Sample')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('C:/Users/Shash/OneDrive/Desktop/GIT_Projects/sentiment-analysis/sentiment_results.png', dpi=150)
    plt.show()

def main():
    analyzer = SentimentAnalyzer()
    
    accuracy = analyzer.train()
    
    test_reviews = [
        "I absolutely loved this movie great acting",
        "Terrible film waste of time",
        "Good movie with excellent storyline",
        "Very boring and bad experience",
        "Best movie ever highly recommend",
        "Horrible acting not worth watching",
        "Amazing film loved every minute",
        "Disappointing movie poor script",
    ]
    
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE REVIEWS")
    print("=" * 60)
    
    for review in test_reviews:
        result = analyzer.predict(review)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    visualize_results(analyzer, test_reviews)
    
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS PROJECT COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()