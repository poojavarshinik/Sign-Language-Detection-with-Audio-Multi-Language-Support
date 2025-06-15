import os
import cv2
# Using absolute path for data directory
DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
number_of_classes = 36
dataset_size = 150
# Try different camera indices until it works
cap = None
for i in range(3):  # Test indices 0, 1, and 2
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
else:
    print("No camera found. Please check your connection.")
    exit()
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print('Collecting data for class {}'.format(j))
    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Start capturing images for the dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Skipping.")
            continue
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        filename = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1
        print(f"Captured image {counter}/{dataset_size} for class {j}")
cap.release()
cv2.destroyAllWindows()
