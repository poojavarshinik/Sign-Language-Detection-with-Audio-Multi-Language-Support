
from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time


prediction_history = []
confidence_threshold = 5  # how many frames to track for confidence
prediction_confidence = 0.0  # default confidence

app = Flask(__name__)

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

suggestion_dict = {
    "A": ["A pleasure to meet you", "All is well, thank you", "Amazing to see you"],
    "B": ["By the way, I’m new here", "Been looking forward to meeting you", "Best regards to you"],
    "C": ["Can I introduce myself?", "Call me Kabilan", "Could you tell me about yourself?"],
    "D": ["Do you know about our project?", "Delighted to meet you", "Don’t hesitate to ask"],
    "E": ["Excited to be here", "Everyone calls me Kabilan", "Eager to collaborate with you"],
    "F": ["Feel free to share your thoughts", "Fantastic to see you", "First time meeting you?"],
    "G": ["Good morning, how are you?", "Glad to meet you", "Greetings to you"],
    "H": ["Hi, my name is Kabilan", "How are you today?", "Hope we can work together"],
    "I": ["I’m from Coimbatore", "I study AI and Data Science", "It’s a pleasure to meet you"],
    "J": ["Just wanted to say hello", "Joy to meet you", "Join me for a chat"],
    "K": ["Keep in touch", "Kind regards", "Know anything about AI?"],
    "L": ["Let me introduce myself", "Looking forward to this meeting", "Lovely to meet you"],
    "M": ["My name is Kabilan", "May I ask your name?", "Meeting you is wonderful"],
    "N": ["Nice to meet you", "New to this field?", "Noticed you earlier, hello"],
    "O": ["Oh, nice to meet you", "Our paths crossed today", "On a journey together"],
    "P": ["Pleasure meeting you", "Pleased to introduce myself", "Proud to meet you"],
    "Q": ["Quite excited to meet you", "Quick intro: I’m Kabilan", "Questions about me?"],
    "R": ["Really glad to meet you", "Ready to learn from you", "Reaching out to say hi"],
    "S": ["So nice to meet you", "Studying AI and Data Science", "Such a great moment"],
    "T": ["Thank you for your time", "Thrilled to meet you", "This is Kabilan speaking"],
    "U": ["Understanding your background is interesting", "Useful to know about you", "Until next time, goodbye"],
    "V": ["Very happy to meet you", "Valued your time today", "Visit Coimbatore sometime"],
    "W": ["What’s your name?", "Wonderful to meet you", "Wishing you a great day"],
    "X": ["Excited to work with you", "Exceptional to meet you", "X marks this moment"],
    "Y": ["Your presence is appreciated", "Yes, I’m Kabilan", "You’re amazing to meet"],
    "Z": ["Zealous about AI projects", "Zero hesitation in meeting you", "Zestful to be here"]
}

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)



latest_prediction = ""
current_word = ""
letter_buffer = []
current_sentence = ""
used_suggestions = {}
suggested_words = []
chosen_index = 0
last_letter = ""
last_time = time.time()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def process_letter(letter):
    global latest_prediction, letter_buffer, current_word, last_letter, last_time, suggested_words, chosen_index
    current_time = time.time()
    if current_time - last_time < 1.5 or letter == last_letter:
        return
    last_time = current_time
    last_letter = letter
    latest_prediction = letter
    letter_buffer.append(letter)
    current_word = ''.join(letter_buffer)

    if letter.isalpha():
        all_suggestions = suggestion_dict.get(letter.upper(), [])
        if letter not in used_suggestions:
            used_suggestions[letter] = []
        remaining = [s for s in all_suggestions if s not in used_suggestions[letter]]
        if not remaining:
            used_suggestions[letter] = []
            remaining = all_suggestions
        suggested_words[:] = remaining
        chosen_index = 0
    else:
        suggested_words[:] = []
        chosen_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_prediction')
# def get_prediction():
#     suggestion = suggested_words[chosen_index] if suggested_words else "No suggestions available"
#     return jsonify({
#         "prediction": latest_prediction,
#         "word": current_word,
#         "suggestion": suggestion
#     })
@app.route('/get_prediction')
def get_prediction():
    suggestion = suggested_words[chosen_index] if suggested_words else "No suggestions available"
    return jsonify({
        "prediction": latest_prediction,
        "confidence": prediction_confidence,
        "word": current_word,
        "suggestion": suggestion
    })

@app.route('/key_input/<key>', methods=['POST'])
def handle_key(key):
    global current_word, letter_buffer, current_sentence, suggested_words, chosen_index
    if key == 'space':
        current_sentence += " "
    elif key == 'c':  # Start new word
        current_word = ""
        letter_buffer = []
    elif key == 'v':  # End word
        current_sentence += current_word + " "
        current_word = ""
        letter_buffer = []
    elif key == ' ':  # Space
        current_sentence += " "
    elif key == 'x':  # Speak suggestion
        if suggested_words:
            speak(suggested_words[chosen_index])
    elif key == 'y':  # Speak current word
        if current_word:
            speak(current_word)
    elif key == 'z':  # Skip suggestion
        if suggested_words:
            used_suggestions.setdefault(last_letter, []).append(suggested_words[chosen_index])
            chosen_index = (chosen_index + 1) % len(suggested_words)
    return '', 204

@app.route('/clear', methods=['POST'])
def clear():
    global current_word, letter_buffer
    current_word = ""
    letter_buffer = []
    return '', 204

def update_prediction(new_prediction):
    global prediction_history, prediction_confidence
    prediction_history.append(new_prediction)
    if len(prediction_history) > confidence_threshold:
        prediction_history.pop(0)

    most_common = max(set(prediction_history), key=prediction_history.count)
    confidence = prediction_history.count(most_common) / len(prediction_history)
    prediction_confidence = confidence
    return most_common

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                data_aux = []
                for i in range(21):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))
                try:
                    # prediction = model.predict([np.asarray(data_aux)])
                    # predicted_character = labels_dict[int(prediction[0])]
                    # process_letter(predicted_character)
                    prediction = model.predict([np.asarray(data_aux)])
                    raw_prediction = labels_dict[int(prediction[0])]
                    stable_prediction = update_prediction(raw_prediction)
                    latest_prediction = stable_prediction
                    process_letter(stable_prediction)

                except Exception as e:
                    print("Prediction error:", e)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
