import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras.models import load_model

# Load the model and encoders
model = load_model('sign_language_model.h5')
encoder = joblib.load('label_encoder.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess landmarks for model prediction
def preprocess_landmarks(landmarks):
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    return landmarks_array.reshape(1, 21, 3)  # Reshape to match model input shape (1, 21, 3)

# Function to get gesture name from prediction
def get_gesture_name(prediction):
    label_encoded = np.argmax(prediction, axis=1)
    return encoder.inverse_transform(label_encoded)[0]

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to get hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess landmarks
            landmarks_array = preprocess_landmarks(hand_landmarks.landmark)
            
            # Predict the gesture
            prediction = model.predict(landmarks_array)
            gesture_name = get_gesture_name(prediction)
            
            # Display the gesture name
            cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
