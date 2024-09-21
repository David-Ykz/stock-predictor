from dotenv import load_dotenv
import os
from newsapi import NewsApiClient
import json
load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

acceptedSources = [
    'abc-news', 
    'bbc-news',
    'bloomberg',
    'business-insider',
    'cbc-news',
    'financial-post',
    'fortune',
    'google-news',
    'hacker-news',
    'reuters',
    'techcrunch',
    'the-wall-street-journal'
]
sourcesAsStr = ""
for source in acceptedSources:  
    sourcesAsStr += source + ","
sourcesAsStr = sourcesAsStr[:-1]
print(sourcesAsStr)

articles = newsapi.get_everything(q="Microsoft", sources="abc-news,bbc-news,bloomberg,business-insider,cbc-news,financial-post,fortune,google-news,hacker-news,reuters,techcrunch,the-wall-street-journal", from_param='2024-09-01', to='2024-09-09',)

with open("articles.json", "w") as json_file:
    json.dump(articles, json_file, indent=4)

print(articles)