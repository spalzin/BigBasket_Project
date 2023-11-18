import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load data
data = pd.read_csv(r'C:\Users\Spalzin\Downloads\BigBasket_Project\bigBasketProducts.csv')   

# Ensure 'description' column is clean (no missing values)
data['description'] = data['description'].fillna('')

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['description'])

# Save the TF-IDF Matrix and Vectorizer for later use
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


