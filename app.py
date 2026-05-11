import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
import json
import tempfile
import time
from tensorflow.image import resize

# ─── Page Config ───
st.set_page_config(
    page_title="SoundSense AI · Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Genre Metadata ───
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_ICONS = {
    'blues': '🎷', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
    'hiphop': '🎤', 'jazz': '🎺', 'metal': '🤘', 'pop': '⭐',
    'reggae': '🌴', 'rock': '🎸'
}
GENRE_COLORS = {
    'blues': '#4A90D9', 'classical': '#D4AF37', 'country': '#E8A838',
    'disco': '#E040FB', 'hiphop': '#FF5722', 'jazz': '#FFB74D',
    'metal': '#B0BEC5', 'pop': '#FF4081', 'reggae': '#66BB6A', 'rock': '#EF5350'
}
GENRE_IMAGES = {
    'blues': 'https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=600&q=80',
    'classical': 'https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=600&q=80',
    'country': 'https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=600&q=80',
    'disco': 'https://images.unsplash.com/photo-1574391884720-bbc3740c59d1?w=600&q=80',
    'hiphop': 'https://images.unsplash.com/photo-1571609803939-54f463c1752d?w=600&q=80',
    'jazz': 'https://images.unsplash.com/photo-1415201364774-f6f0bb35f28f?w=600&q=80',
    'metal': 'https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=600&q=80',
    'pop': 'https://images.unsplash.com/photo-1501386761578-eac5c94b800a?w=600&q=80',
    'reggae': 'https://images.unsplash.com/photo-1530651788726-1dbf58eeef1f?w=600&q=80',
    'rock': 'https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=600&q=80',
}
GENRE_DESC = {
    'blues': 'Soulful melodies rooted in African-American traditions.',
    'classical': 'Timeless orchestral compositions of Western art music.',
    'country': 'Storytelling through acoustic strings and heartfelt lyrics.',
    'disco': 'Groovy, danceable beats from the golden era of nightlife.',
    'hiphop': 'Rhythmic poetry over beats, born from urban culture.',
    'jazz': 'Improvisational artistry with rich harmonic complexity.',
    'metal': 'High-energy, distorted guitars and powerful vocals.',
    'pop': 'Catchy hooks and universal appeal for the masses.',
    'reggae': 'Offbeat rhythms carrying messages of love and resistance.',
    'rock': 'Raw energy driven by electric guitars and drums.'
}

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg-primary: #0a0a0f; --bg-secondary: #12121a; --bg-card: rgba(255,255,255,0.04);
    --text-primary: #f0f0f5; --text-secondary: #a0a0b0;
    --accent: #7c3aed; --accent-glow: rgba(124,58,237,0.3);
    --glass-bg: rgba(255,255,255,0.05); --glass-border: rgba(255,255,255,0.08);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1025 50%, #0a0a0f 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d15 0%, #15102a 100%) !important;
    border-right: 1px solid var(--glass-border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
.glass-card {
    background: var(--glass-bg); border: 1px solid var(--glass-border);
    border-radius: 20px; padding: 30px; backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 20px;
    transition: transform 0.3s, box-shadow 0.3s;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(124,58,237,0.15); }
.hero-section {
    text-align: center; padding: 60px 20px;
    background: radial-gradient(ellipse at center, rgba(124,58,237,0.12) 0%, transparent 70%);
    border-radius: 24px; margin-bottom: 30px;
}
.hero-title {
    font-size: 3.2rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(135deg, #7c3aed, #a78bfa, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.hero-sub { color: var(--text-secondary); font-size: 1.15rem; font-weight: 300; }
.genre-badge {
    display: inline-block; padding: 6px 18px; border-radius: 50px;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;
}
.stat-value { font-size: 2rem; font-weight: 800; color: #a78bfa; }
.stat-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }
.confidence-bar-bg {
    background: rgba(255,255,255,0.06); border-radius: 10px; height: 28px;
    overflow: hidden; margin: 6px 0;
}
.confidence-bar-fill {
    height: 100%; border-radius: 10px; display: flex; align-items: center;
    padding-left: 12px; font-size: 0.75rem; font-weight: 700; color: #fff;
    transition: width 0.8s ease;
}
.section-title {
    font-size: 1.4rem; font-weight: 700; color: var(--text-primary);
    margin-bottom: 16px; display: flex; align-items: center; gap: 10px;
}
.upload-zone {
    border: 2px dashed rgba(124,58,237,0.4); border-radius: 20px;
    padding: 40px; text-align: center; background: rgba(124,58,237,0.03);
    transition: border-color 0.3s, background 0.3s;
}
.upload-zone:hover { border-color: rgba(124,58,237,0.7); background: rgba(124,58,237,0.06); }
.tech-chip {
    display: inline-block; padding: 4px 14px; border-radius: 50px;
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);
    color: #a78bfa; font-size: 0.78rem; font-weight: 500; margin: 3px;
}
.result-genre {
    font-size: 2.8rem; font-weight: 900; text-align: center; margin: 10px 0;
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
h1, h2, h3, h4, h5, h6, p, span, div, label { color: var(--text-primary); }
div[data-testid="stFileUploader"] > div { border: none !important; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 10px 28px !important; font-weight: 600 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ───
@st.cache_resource
def load_model():
    model_path = "music_genre_classification_model.keras"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return tf.keras.models.load_model(model_path)

# ─── Audio Preprocessing ───
def load_and_preprocess_audio(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration, chunk_overlap = 4, 2
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(chunk_overlap * sample_rate)
    step = chunk_samples - overlap_samples
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk duration")
    if len(audio_data) < chunk_samples:
        return np.array(data)
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / step)) + 1
    for i in range(num_chunks):
        start = i * step
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            continue
        S = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        S_resized = resize(np.expand_dims(S, axis=-1), target_shape)
        data.append(S_resized)
    return np.array(data)

# ─── Prediction ───
def predict_genre(model, file_path):
    X = load_and_preprocess_audio(file_path)
    if len(X) == 0:
        return None, None
    preds = model.predict(X, verbose=0)
    avg_preds = np.mean(preds, axis=0)
    return GENRES[np.argmax(avg_preds)], avg_preds

# ─── Sidebar ───
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:20px 0 10px;">'
                '<span style="font-size:2.5rem;">🎵</span><br>'
                '<span style="font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#7c3aed,#a78bfa);'
                '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">SoundSense AI</span></div>',
                unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🎯 Predict", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")

    # Model Info
    st.markdown('<div class="section-title">📊 Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="padding:16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#a0a0b0;font-size:0.82rem;">Architecture</span>
            <span style="color:#a78bfa;font-weight:600;font-size:0.82rem;">CNN</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#a0a0b0;font-size:0.82rem;">Genres</span>
            <span style="color:#a78bfa;font-weight:600;font-size:0.82rem;">10</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#a0a0b0;font-size:0.82rem;">Val Accuracy</span>
            <span style="color:#66BB6A;font-weight:600;font-size:0.82rem;">90.02%</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#a0a0b0;font-size:0.82rem;">Input</span>
            <span style="color:#a78bfa;font-weight:600;font-size:0.82rem;">Mel Spec 150×150</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # History
    if 'history' not in st.session_state:
        st.session_state.history = []
    if st.session_state.history:
        st.markdown('<div class="section-title">🕒 History</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-5:]):
            icon = GENRE_ICONS.get(h['genre'], '🎵')
            color = GENRE_COLORS.get(h['genre'], '#a78bfa')
            st.markdown(f'<div style="padding:8px 12px;margin-bottom:6px;border-radius:10px;'
                        f'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
                        f'font-size:0.8rem;">{icon} <span style="color:{color};font-weight:600;">'
                        f'{h["genre"].title()}</span><br><span style="color:#666;font-size:0.7rem;">'
                        f'{h["file"]}</span></div>', unsafe_allow_html=True)

# ─── HOME PAGE ───
if "Home" in page:
    st.markdown("""
    <div class="hero-section">
        <div style="font-size:4rem;margin-bottom:10px;">🎧</div>
        <div class="hero-title">SoundSense AI</div>
        <div class="hero-sub">Intelligent Music Genre Classification powered by Deep Learning</div>
        <div style="margin-top:20px;">
            <span class="tech-chip">TensorFlow</span>
            <span class="tech-chip">Mel Spectrograms</span>
            <span class="tech-chip">CNN</span>
            <span class="tech-chip">Librosa</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    stats = [("10", "Music Genres"), ("90%", "Accuracy"), ("< 5s", "Analysis Time")]
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:24px;">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎵 How It Works</div>', unsafe_allow_html=True)
    steps = [("🎙️ Upload", "Drop your audio file (WAV, MP3, OGG) into the analyzer."),
             ("📊 Process", "Audio is split into chunks and converted to Mel Spectrograms."),
             ("🧠 Classify", "Our CNN model analyzes spectral patterns across all chunks."),
             ("🎯 Result", "Get the predicted genre with confidence scores.")]
    step_cols = st.columns(4)
    for col, (icon, desc) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;min-height:160px;">
                <div style="font-size:2rem;margin-bottom:10px;">{icon}</div>
                <div style="font-weight:600;margin-bottom:6px;color:#e0e0f0;">{icon.split()[0] if len(icon)>2 else ''}</div>
                <div style="color:#a0a0b0;font-size:0.85rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎼 Supported Genres</div>', unsafe_allow_html=True)
    g_cols = st.columns(5)
    for i, genre in enumerate(GENRES):
        with g_cols[i % 5]:
            color = GENRE_COLORS[genre]
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:18px;min-height:90px;">
                <div style="font-size:1.8rem;">{GENRE_ICONS[genre]}</div>
                <div style="font-weight:700;color:{color};margin-top:6px;font-size:0.9rem;">{genre.title()}</div>
            </div>""", unsafe_allow_html=True)

# ─── PREDICT PAGE ───
elif "Predict" in page:
    st.markdown("""
    <div style="text-align:center;margin-bottom:30px;">
        <span style="font-size:2.5rem;">🎯</span>
        <h2 style="font-weight:800;margin:8px 0 4px;background:linear-gradient(135deg,#7c3aed,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Genre Prediction</h2>
        <p style="color:#a0a0b0;">Upload your audio and let AI identify the genre</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_link = st.columns(2)
    with col_upload:
        st.markdown('<div class="glass-card"><div class="section-title">📁 Upload Audio</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop your audio file here", type=["wav", "mp3", "ogg"],
                                     label_visibility="collapsed", key="audio_upload")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_link:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🔗 Paste Music URL</div>
            <p style="color:#a0a0b0;font-size:0.85rem;">URL support coming soon. Use file upload for now.</p>
        </div>""", unsafe_allow_html=True)
        url_input = st.text_input("Paste a music URL", placeholder="https://example.com/song.mp3",
                                   label_visibility="collapsed", disabled=True)

    if uploaded is not None:
        st.markdown("---")
        # Audio Player
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎧 Audio Preview</div>', unsafe_allow_html=True)
        st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Analyze Genre", use_container_width=True):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            # Loading animation
            spinner_msgs = [
                "🎵 Loading audio waveform...",
                "📊 Extracting Mel Spectrograms...",
                "🧠 Analyzing rhythm patterns...",
                "🎶 Decoding melody structure...",
                "🎯 Classifying genre..."
            ]
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, msg in enumerate(spinner_msgs):
                status_text.markdown(f'<div style="text-align:center;color:#a78bfa;font-weight:500;'
                                     f'font-size:0.95rem;" class="pulse">{msg}</div>',
                                     unsafe_allow_html=True)
                progress_bar.progress((i + 1) * 15)
                time.sleep(0.4)

            try:
                model = load_model()
                progress_bar.progress(80)
                genre, confidences = predict_genre(model, tmp_path)
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                if genre is None:
                    st.error("⚠️ Audio too short. Please upload a longer clip (>4 seconds).")
                else:
                    # Save to history
                    st.session_state.history.append({"genre": genre, "file": uploaded.name})

                    color = GENRE_COLORS[genre]
                    icon = GENRE_ICONS[genre]
                    img = GENRE_IMAGES[genre]
                    desc = GENRE_DESC[genre]
                    conf_pct = confidences[np.argmax(confidences)] * 100

                    # Result Card
                    st.markdown(f"""
                    <div class="glass-card" style="border:1px solid {color}33;
                         background:linear-gradient(135deg, {color}08, transparent);">
                        <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;">
                            <div style="flex:0 0 200px;">
                                <img src="{img}" style="width:200px;height:200px;object-fit:cover;
                                     border-radius:16px;border:2px solid {color}44;" />
                            </div>
                            <div style="flex:1;min-width:250px;">
                                <div style="font-size:0.85rem;color:#a0a0b0;text-transform:uppercase;
                                     letter-spacing:2px;font-weight:600;">Detected Genre</div>
                                <div class="result-genre">{icon} {genre.upper()}</div>
                                <div style="color:#a0a0b0;font-size:0.9rem;margin-bottom:12px;">{desc}</div>
                                <div class="genre-badge" style="background:{color}22;color:{color};
                                     border:1px solid {color}44;">
                                    Confidence: {conf_pct:.1f}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence Bars
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Confidence Breakdown</div>',
                                unsafe_allow_html=True)

                    sorted_idx = np.argsort(confidences)[::-1]
                    bars_html = ""
                    for idx in sorted_idx:
                        g = GENRES[idx]
                        c = confidences[idx] * 100
                        gc = GENRE_COLORS[g]
                        gi = GENRE_ICONS[g]
                        w = max(c, 2)
                        bars_html += f"""
                        <div style="display:flex;align-items:center;margin-bottom:8px;gap:10px;">
                            <div style="width:90px;font-size:0.8rem;color:#d0d0e0;font-weight:500;
                                 text-align:right;">{gi} {g.title()}</div>
                            <div style="flex:1;">
                                <div class="confidence-bar-bg">
                                    <div class="confidence-bar-fill"
                                         style="width:{w}%;background:linear-gradient(90deg,{gc},{gc}88);">
                                        {c:.1f}%
                                    </div>
                                </div>
                            </div>
                        </div>"""
                    st.markdown(bars_html + '</div>', unsafe_allow_html=True)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error during analysis: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ─── ABOUT PAGE ───
elif "About" in page:
    st.markdown("""
    <div style="text-align:center;margin-bottom:30px;">
        <span style="font-size:2.5rem;">ℹ️</span>
        <h2 style="font-weight:800;margin:8px 0 4px;background:linear-gradient(135deg,#7c3aed,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">About SoundSense AI</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🧠 The Technology</div>
            <p style="color:#a0a0b0;font-size:0.9rem;line-height:1.7;">
                SoundSense AI uses a <b style="color:#a78bfa;">Convolutional Neural Network (CNN)</b>
                trained on the GTZAN dataset. Audio is split into overlapping 4-second chunks,
                converted to <b style="color:#a78bfa;">Mel Spectrograms</b>, resized to 150×150,
                and fed through the model. Predictions are averaged across all chunks for robust results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">📋 Tech Stack</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;">
                <span class="tech-chip">Python</span>
                <span class="tech-chip">TensorFlow / Keras</span>
                <span class="tech-chip">Librosa</span>
                <span class="tech-chip">Streamlit</span>
                <span class="tech-chip">NumPy</span>
                <span class="tech-chip">Mel Spectrograms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">📈 Model Performance</div>
            <p style="color:#a0a0b0;font-size:0.9rem;line-height:1.7;">
                Trained for <b style="color:#a78bfa;">30 epochs</b> achieving
                <b style="color:#66BB6A;">~90% validation accuracy</b>. The model generalizes
                well across all 10 genres with balanced precision and recall.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Training curve
        try:
            hist_path = os.path.join(os.path.dirname(__file__), 'training_history.json')
            with open(hist_path) as f:
                hist = json.load(f)
            st.markdown('<div class="glass-card"><div class="section-title">📉 Training Curves</div>',
                        unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame({
                "Train Accuracy": hist["accuracy"],
                "Val Accuracy": hist["val_accuracy"]
            })
            st.line_chart(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            pass

# ─── Footer ───
st.markdown("""
<div style="text-align:center;padding:40px 0 20px;border-top:1px solid rgba(255,255,255,0.05);
     margin-top:40px;">
    <span style="color:#555;font-size:0.8rem;">
        Built with ❤️ using Streamlit & TensorFlow · SoundSense AI © 2026
    </span>
</div>
""", unsafe_allow_html=True)