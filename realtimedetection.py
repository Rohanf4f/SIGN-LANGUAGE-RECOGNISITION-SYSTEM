import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json

# Load your trained model
json_file = open("signlanguagedetectionmodel28x128.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel28x128.h5")

# Function to extract features from the image
def extract_features(image):
    # Ensure image has 3 channels (convert grayscale to RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    feature = np.array(image_rgb)
    feature = feature.reshape(1, 128, 128, 3)
    return feature / 255.0

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define the labels for your classes
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally to avoid mirroring
    frame = cv2.flip(frame, 1)
    
    # Convert the image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands in the frame using MediaPipe
    results = hands.process(frame_rgb)
    
    # If hands are detected, extract landmarks and draw bounding box
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand region based on detected landmarks
            img_height, img_width, _ = frame.shape
            landmark_list = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                landmark_list.append((x, y))
            
            # Get the bounding box of the hand
            xmin = min([x for x, y in landmark_list])
            xmax = max([x for x, y in landmark_list])
            ymin = min([y for x, y in landmark_list])
            ymax = max([y for x, y in landmark_list])
            
            # Draw bounding box around the hand
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Crop the hand region
            hand_crop = frame[ymin:ymax, xmin:xmax]
            if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                hand_gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                hand_resized = cv2.resize(hand_gray, (128, 128))
                hand_features = extract_features(hand_resized)
                
                # Predict the sign language gesture using your trained model
                pred = model.predict(hand_features)
                prediction_label = label[pred.argmax()]
                accu = "{:.2f}".format(np.max(pred)*100)
                if prediction_label == 'blank':
                    cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the output frame
    cv2.imshow('Sign Language Detection', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
