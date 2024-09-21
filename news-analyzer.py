import json
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis")

with open('articles.json', 'r') as file:
    data = json.load(file)

for article in data.get('articles'):
    title = article.get('title')
    if title:
        result = sentiment_analysis(title)[0]
        label = result['label']
        score = result['score']
        print(f"Title: {title}\nSentiment: {label}, Confidence: {score:.2f}\n")
