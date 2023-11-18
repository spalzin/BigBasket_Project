import joblib
from sklearn.metrics.pairwise import linear_kernel
import joblib
from sklearn.metrics.pairwise import linear_kernel



tfidf_matrix = joblib.load('tfidf_matrix.pkl')
vectorizer = joblib.load('vectorizer.pkl')
def get_similar_products(query, top_n=5):
    query_vector = vectorizer.transform([query])
    cosine_similarities_query = linear_kernel(query_vector, tfidf_matrix).flatten()
    similar_product_indices = cosine_similarities_query.argsort()[:-top_n-1:-1]
    return similar_product_indices


