from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def content_based_filtering(data):
    tfidf = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf.fit_transform(data['review_title'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim
