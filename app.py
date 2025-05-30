from transformers import pipeline
from deep_translator import GoogleTranslator
import requests
import gradio as gr
import random

# 🎯 載入情緒模型
emotion_model = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None
)

# 🔑 Last.fm API 金鑰
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

# 🎯 主邏輯
def recommend(text, style):
    emotion = None
    confidence = None
    emotion_tag = ""
    style_tag = style.strip().lower() if style else ""

    if text.strip():
        translated = translate_to_english(text)
        emotion, confidence = detect_emotion(translated)
        emotion_tag = map_emotion_to_tag(emotion)

    tag_parts = [tag for tag in [emotion_tag, style_tag] if tag]
    if not tag_parts:
        return "⚠️ Please enter at least one mood or select a style."

    final_tag = " ".join(tag_parts)

    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        'method': 'tag.gettoptracks',
        'tag': final_tag,
        'api_key': LASTFM_API_KEY,
        'format': 'json',
        'limit': 30
    }

    response = requests.get(url, params=params)
    data = response.json()
    tracks = data.get('tracks', {}).get('track', [])
    random.shuffle(tracks)
    tracks = tracks[:5]

    if not tracks:
        return f"🫤 No matching songs found (tag: {final_tag})"

    songs = "\n".join([f"🎵 {t['name']} - {t['artist']['name']}" for t in tracks])
    emotion_info = f"Emotion: {emotion} (confidence: {confidence})\n" if emotion else ""
    return f"{emotion_info}\n🎶 Recommended songs:\n{songs}"

# 🎛️ Gradio Blocks UI
with gr.Blocks() as interface:
    gr.Markdown("# 🎧 Mood and style music recommendation")
    gr.Markdown("You can enter only the mood, only the genre, or both, and AI will randomly recommend 5 songs")

    mood = gr.Textbox(label="🧠 Please enter your mood (optional)")
    genre = gr.Dropdown(
        choices=["", "rock", "pop", "jazz", "hip-hop", "electronic", "classical", "chill"],
        label="🎼 Select a style (optional)",
        value=""
    )
    output = gr.Textbox(label="🎵 AI recommendation results")
    button = gr.Button("Recommend")

    button.click(fn=recommend, inputs=[mood, genre], outputs=output)

interface.queue()
interface.launch()
