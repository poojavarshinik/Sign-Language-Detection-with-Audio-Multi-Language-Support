
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

# Load modelw
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Initialize camera
cap = None
camera_found = False
for i in range(3):  # Test camera indices 0, 1, 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        camera_found = True
        break

if not camera_found:
    print("No camera found. Please check your connection.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

# Predefined suggestions
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

letter_buffer = []
current_word = ""
suggested_words = []
chosen_index = 0
used_suggestions = {}  # Tracks used suggestions for each letter

# Timer setup
start_time = time.time()
max_duration = 120  # 2 minutes in seconds
last_stored_time = time.time()

def process_recognized_letter(letter):
    global letter_buffer, current_word, suggested_words, chosen_index, last_stored_time, used_suggestions
    current_time = time.time()

    if current_time - last_stored_time >= 2:
        if letter == " " or letter == "END":
            if letter_buffer:
                word = ''.join(letter_buffer)
                current_word = word
                letter_buffer = []
        else:
            letter_buffer.append(letter)
            current_word = ''.join(letter_buffer)

            # Provide suggestions if available
            if letter not in used_suggestions:
                used_suggestions[letter] = []  # Initialize for new letter

            all_suggestions = suggestion_dict.get(letter, [])
            suggested_words = [s for s in all_suggestions if s not in used_suggestions[letter]]

            if not suggested_words:
                suggested_words = all_suggestions  # Reset if all suggestions have been used
                used_suggestions[letter] = []     # Reset used list for the letter

            chosen_index = 0  # Reset to the first suggestion
        last_stored_time = current_time

# Create a window to display the word
cv2.namedWindow("Word Display", cv2.WINDOW_NORMAL)

while cap.isOpened():
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for i in range(21):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            process_recognized_letter(predicted_character)
        except Exception as e:
            print("Error during prediction:", e)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
    word_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(word_frame, f"Word: {current_word}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if suggested_words:
        cv2.putText(word_frame, f"Suggestion: {suggested_words[chosen_index]}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(word_frame, "(Press 'x' to change, 'y' to confirm)", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Word Display", word_frame)
    cv2.imshow('Sign Language Detector', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('w') or (time.time() - start_time) > max_duration:
        print("Exiting program after pressing 'w' or 2 minutes.")
        break
    elif key == ord(' '):  # Add space when spacebar is pressed
        if letter_buffer:
            letter_buffer.append(' ')
            current_word = ''.join(letter_buffer)
    elif key == ord('x') and suggested_words:   # Change suggestion
        used_suggestions[current_word[-1]].append(suggested_words[chosen_index])
        chosen_index = (chosen_index + 1) % len(suggested_words)
    elif key == ord('y') and suggested_words:  # Confirm suggestion4
        current_word = suggested_words[chosen_index]
        print(f"Final word chosen: {current_word}")
        break
cap.release()
cv2.destroyAllWindows()
print("Final recognized word: ", current_word)
engine.say(current_word)
engine.runAndWait()
