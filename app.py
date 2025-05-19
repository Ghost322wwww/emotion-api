from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import gradio as gr

# 🎯 載入情緒模型
emotion_model = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None
)

# 🔑 Last.fm API 金鑰（正式請改環境變數）
LASTFM_API_KEY = "ed823d59db776e6c09055d838788e9fe"

# 🌐 中文翻譯
def translate_to_english(text):
    return GoogleTranslator(source='zh-TW', target='en').translate(text)

# 🧠 情緒偵測
def detect_emotion(text):
    results = emotion_model(text)[0]
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_results[0]['label']
    confidence = round(sorted_results[0]['score'], 2)
    return top_emotion, confidence

# 🎵 對應情緒到 Last.fm tag
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

# 🎯 主邏輯：輸入文字 → 回傳情緒 + 推薦歌曲
def recommend(text):
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
        return f"Emotion: {emotion} (confidence: {confidence})\n\n😕 沒有找到歌曲"

    songs = "\n".join([f"🎵 {t['name']} - {t['artist']['name']}" for t in tracks])
    return f"Emotion: {emotion} (confidence: {confidence})\n\n🎶 推薦歌曲：\n{songs}"

# 🎛️ Gradio UI
interface = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="🧠 Please enter your mood (Chinese is acceptable)"),
    outputs=gr.Textbox(label="🎵 AI recommendation results"),
    title="🎧 AI music mood recommendation",
    description="Enter your mood, AI will judge your mood and recommend songs (Chinese is also OK)"
)
interface.launch()
