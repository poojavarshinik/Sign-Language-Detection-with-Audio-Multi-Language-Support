# Sign-Language-Detection-with-Audio-Multi-Language-Support

This project is a real-time sign language recognition system that uses a webcam to detect hand gestures and convert them into letters, words, and meaningful phrases. Suggested text is spoken aloud using text-to-speech and can also be translated into multiple languages using an integrated LLM-based translation backend.

---

## 🌟 Features

- 🔤 **Real-time Gesture Recognition**: Detects hand gestures using MediaPipe and classifies them with a trained Random Forest model.
- 💬 **Phrase Suggestions**: Each recognized letter triggers pre-defined contextual phrases (e.g., “A” → “Amazing to see you”).
- 🔊 **Text-to-Speech Output**: Converts recognized words and phrases to speech using `pyttsx3`.
- 🌍 **Multilingual Support**: Suggested phrases are automatically translated into other languages (e.g., Tamil, Hindi, French) via LLM APIs in the backend.
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
| `app.py` | Flask app serving live video, predictions, phrase suggestions, TTS, and language translation |
| `collect_imgs.py` | Collect gesture data for training (0–9, A–Z) |
| `create_dataset.py` | Converts raw images to normalized hand landmark vectors |
| `train_classifier.py` | Trains the RandomForest classifier and saves the model |
| `inference_classifier.py` | Real-time classifier (CLI) with visual and TTS feedback |
| `model.p` | Trained gesture recognition model |
| `data.pickle` | Processed dataset (landmarks and labels) |
| `templates/index.html` | UI page for the web app (served by Flask) |

---

