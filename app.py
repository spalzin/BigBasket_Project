from flask import Flask, request, jsonify
from llm import get_similar_products

app = Flask(__name__)

@app.route('/get_similar_products', methods=['POST'])
def get_similar_products_api():
    data = request.get_json()
    query = data['query']
    top_n = data.get('top_n', 5)
    similar_products = get_similar_products(query, top_n)
    return jsonify({'similar_products': list(similar_products)})

if __name__ == '__main__':
    app.run(debug=True)
