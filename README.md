# 🎵 SoundSense AI — Music Genre Classification

SoundSense AI is a deep learning powered music genre classification web application built using Streamlit, TensorFlow, and Librosa. The application analyzes uploaded audio files, converts them into Mel Spectrograms, and predicts the most likely music genre using a Convolutional Neural Network (CNN).

---

# 🚀 Features

* 🎧 Upload and analyze audio files instantly
* 🧠 CNN-based deep learning model for genre prediction
* 📊 Confidence score visualization for all genres
* 🎼 Supports 10 music genres
* ✨ Modern glassmorphism UI with custom Streamlit styling
* 📈 Training curve visualization
* 🕒 Prediction history tracking
* ⚡ Fast and responsive inference pipeline

---

# 🎯 Supported Genres

| Genre     | Icon |
| --------- | ---- |
| Blues     | 🎷   |
| Classical | 🎻   |
| Country   | 🤠   |
| Disco     | 🪩   |
| HipHop    | 🎤   |
| Jazz      | 🎺   |
| Metal     | 🤘   |
| Pop       | ⭐    |
| Reggae    | 🌴   |
| Rock      | 🎸   |

---

# 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) trained on Mel Spectrogram representations of audio.

## Audio Processing Pipeline

1. Audio file uploaded by user
2. Audio split into overlapping 4-second chunks
3. Mel Spectrogram generated using Librosa
4. Spectrogram resized to 150×150
5. CNN predicts probabilities for each chunk
6. Final prediction obtained using average confidence scores

---

# 📊 Model Performance

* Validation Accuracy: ~90%
* Input Shape: 150 × 150 Mel Spectrograms
* Framework: TensorFlow / Keras
* Training Epochs: 30

---

# 🛠️ Tech Stack

## Frontend

* Streamlit
* Custom CSS

## Machine Learning

* TensorFlow
* Keras
* Librosa
* NumPy

## Visualization

* Streamlit Charts
* Mel Spectrograms

---

# 📁 Project Structure

```bash
Music-Genre-Classification/
│
├── app.py
├── music_genre_classification_model.keras
├── training_history.json
├── requirements.txt
├── .gitattributes
├── README.md
└── assets/
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/subham-3282/Music-Genre-Classification.git
cd Music-Genre-Classification
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will start locally at:

```bash
http://localhost:8501
```

---

# 📦 Requirements

Example dependencies:

```txt
streamlit
tensorflow
numpy
librosa
pandas
```

---

# 🖼️ Application Preview

## Home Page

* Modern hero section
* Genre showcase
* Model statistics
* Workflow explanation

## Prediction Page

* Upload audio files
* Audio preview player
* AI genre prediction
* Confidence breakdown bars
* Genre image cards

## About Page

* Tech stack overview
* Model details
* Training curve visualization

---

# 🔥 Future Improvements

* 🔗 Music URL support
* ☁️ Cloud deployment
* 📱 Mobile responsive optimization
* 🎼 More genre support
* 🎙️ Real-time microphone classification
* 🧠 Transformer-based audio models

---

# 🌐 Deployment Options

The project can be deployed using:

* Streamlit Community Cloud
* Render
* Hugging Face Spaces
* Railway
* AWS / GCP / Azure

---

# 📚 Dataset

Model trained using the GTZAN Music Genre Dataset.

Dataset includes:

* 10 genres
* 100 audio tracks per genre
* 30-second audio clips

---

# 👨‍💻 Author

## Subham Sahu

* GitHub: [https://github.com/subham-3282](https://github.com/subham-3282)

---

# ⭐ Acknowledgements

* TensorFlow
* Streamlit
* Librosa
* GTZAN Dataset

---

# 📜 License

This project is licensed under the MIT License.

---

# 💡 Demo Workflow

```text
Upload Audio → Generate Spectrogram → CNN Prediction → Genre Output
```

---

