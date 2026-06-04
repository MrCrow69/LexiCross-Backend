from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim.downloader as api

app = Flask(__name__)
CORS(app)

# Fallback homepage route to test live connectivity
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "healthy", "message": "LexiCross GoogleNews-300 engine is operational!"})

print("⏳ Loading GoogleNews-vectors-negative300 fraction... (This takes about 1-2 minutes)")
# This pulls the optimized 300-dimensional Google News matrix safely into our RAM limits
model = api.load("word2vec-google-news-300")
print("✅ 300-Dimensional Google News Semantic engine is fully loaded and online!")

@app.route('/word2vec/similarity', methods=['GET'])
def get_similarity():
    w1 = request.args.get('w1', '').lower().strip()
    w2 = request.args.get('w2', '').lower().strip()
    
    if not w1 or not w2:
        return jsonify({"error": "Missing words"}), 400
        
    try:
        # Runs the advanced 300-dimensional cosine similarity math
        score = float(model.similarity(w1, w2))
        return jsonify({"similarity": score})
    except KeyError:
        return jsonify({"similarity": 0.0, "not_found": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)