# Sign Language to Speech Assistant with Multilingual Translation

This project is a real-time sign language recognition system that uses a webcam to detect hand gestures and convert them into letters, words, and meaningful phrases. Suggested text is spoken aloud using text-to-speech and also translated into multiple languages using an integrated LLM-based translation backend.

---

## 🌟 Features

- 🔤 **Real-time Gesture Recognition**: Detects hand gestures using MediaPipe and classifies them with a trained Random Forest model.
- 💬 **Phrase Suggestions**: Each recognized letter triggers pre-defined contextual phrases (e.g., “A” → “Amazing to see you”).
- 🔊 **Text-to-Speech Output**: Converts recognized words and phrases to speech using `pyttsx3`.
- 🌍 **Multilingual Support**: Suggested phrases are automatically translated into other languages (e.g., Tamil, Hindi, French) via integrated LLM APIs.
- 🧠 **Smart Confidence Tracking**: Smooths predictions over frames to ensure reliable output.
- ⌨️ **Keyboard Controls**: 
  - `x` – Speak suggested phrase  
  - `y` – Speak current word  
  - `z` – Cycle through suggestions  
  - `c` – Clear current word  
  - `v` – Commit word to sentence  

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `app.py` | Flask app serving webcam feed, predictions, suggestions, TTS, and multilingual output |
| `collect_imgs.py` | Collect gesture data (0–9, A–Z) |
| `create_dataset.py` | Extracts hand landmarks and prepares dataset |
| `train_classifier.py` | Trains and saves the gesture recognition model |
| `inference_classifier.py` | Command-line real-time gesture detection |
| `model.p` | Saved trained Random Forest model |
| `data.pickle` | Processed dataset for training |
| `templates/index.html` | Web interface for real-time interaction |

---

## 🛠 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sign-language-multilang-assistant.git
cd sign-language-multilang-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

Then open your browser and visit:

```
http://localhost:5000
```

---

## 🎓 How It Works

1. **Capture Gesture**: MediaPipe extracts 21 hand landmarks.
2. **Predict Letter**: Random Forest classifier outputs predicted character.
3. **Suggest Phrases**: Based on letter, shows 3 socially relevant suggestions.
4. **Translate Suggestion**: Sends chosen phrase to backend LLM to translate.
5. **Speak Output**: Uses `pyttsx3` to convert text to spoken audio.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `c` | Clear current letter buffer |
| `v` | Finalize word and add to sentence |
| `x` | Speak current suggestion |
| `y` | Speak current word |
| `z` | Skip to next suggestion |

