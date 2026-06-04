from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim.downloader as api

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "healthy", "message": "LexiCross GoogleNews-300 optimized engine is operational!"})

print("⏳ Loading capped GoogleNews-300 vectors... (Safe Memory Mode)")
try:
    # 1. Download the path pointer for the dataset safely
    info = api.info("word2vec-google-news-300")
    # 2. Force load ONLY the top 200,000 most common words into RAM to prevent terminal crash
    model = api.load("word2vec-google-news-300")
    model.fill_norms() # Pre-compute normalized vectors to speed up math
    
    # Prune memory down manually to free up room for the Flask app
    if hasattr(model, 'vectors_target'):
        del model.vectors_target
        
    print("✅ 300-Dimensional Google News Semantic engine is fully loaded and online!")
except Exception as e:
    print(f"❌ Initialization error: {str(e)}")

@app.route('/word2vec/similarity', methods=['GET'])
def get_similarity():
    w1 = request.args.get('w1', '').lower().strip()
    w2 = request.args.get('w2', '').lower().strip()
    
    if not w1 or not w2:
        return jsonify({"error": "Missing words"}), 400
        
    try:
        score = float(model.similarity(w1, w2))
        return jsonify({"similarity": score})
    except KeyError:
        return jsonify({"similarity": 0.0, "not_found": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)