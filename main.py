from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim.downloader as api

app = Flask(__name__)
# This allows your local HTML file to talk to this cloud server securely
CORS(app)

print("⏳ Loading AI word vectors... (This takes about 60 seconds)")
# Using a lightweight 50-dimensional Wikipedia model that loads fast and runs great
model = api.load("glove-wiki-gigaword-50")
print("✅ Semantic engine is fully loaded and online!")

@app.route('/word2vec/similarity', methods=['GET'])
def get_similarity():
    w1 = request.args.get('w1', '').lower().strip()
    w2 = request.args.get('w2', '').lower().strip()
    
    if not w1 or not w2:
        return jsonify({"error": "Missing words"}), 400
        
    try:
        # Math calculation for how close the words are (-1.0 to 1.0)
        score = float(model.similarity(w1, w2))
        return jsonify({"similarity": score})
    except KeyError:
        # If a player types a typo or weird word the AI doesn't know
        return jsonify({"similarity": 0.0, "not_found": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
