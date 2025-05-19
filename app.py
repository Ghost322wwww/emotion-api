from flask import Flask, request, jsonify
from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import os

app = Flask(__name__)

# ========================
# 🔑 Last.fm API Key
# ========================
LASTFM_API_KEY = "ed823d59db776e6c09055d838788e9fe"

# ========================
# 🎯 Emotion Model
# ========================
emotion_model = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None
)


# ========================
# 🌐 Translate Chinese to English
# ========================
def translate_to_english(text):
    translated = GoogleTranslator(source='zh-TW', target='en').translate(text)
    print(f"[DEBUG] Translated input: {translated}")
    return translated

# ========================
# 🧠 Detect Emotion
# ========================
def detect_emotion(text):
    results = emotion_model(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]['label']
    confidence = sorted_results[0]['score']
    return top_emotion, confidence

# ========================
# 🎵 Map Emotion to Last.fm Tag
# ========================
def map_emotion_to_tag(emotion):
    mapping = {
        'joy': 'happy',
        'love': 'romantic',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'dark',
        'surprise': 'upbeat'
    }
    return mapping.get(emotion, 'chill')


# ========================
# 🧠 Flask API Endpoint
# ========================
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text' in request."}), 400

        translated = translate_to_english(text)
        emotion, confidence = detect_emotion(translated)
        tag = map_emotion_to_tag(emotion)

        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'tag.gettoptracks',
            'tag': tag,
            'api_key': LASTFM_API_KEY,
            'format': 'json',
            'limit': 5
        }

        response = requests.get(url, params=params)
        data = response.json()
        tracks = data.get('tracks', {}).get('track', [])

        if not tracks:
            return jsonify({
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "songs": []
            })

        songs = [{"title": t["name"], "artist": t["artist"]["name"]} for t in tracks]

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "songs": songs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# 🚀 Start Server
# ========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)