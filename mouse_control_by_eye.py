import cv2
import numpy as np
import pyautogui
import time

# Constants for screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Load the pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables to store previous eye and face positions
previous_eye_x = None
previous_eye_y = None
previous_face_x = None
previous_face_y = None

# Movement threshold to avoid jittering
MOVEMENT_THRESHOLD = 5

SMOOTHING_WINDOW = 5
eye_positions_x = []
eye_positions_y = []

# Head movement detection threshold
HEAD_MOVEMENT_THRESHOLD = 50

# Blink detection variables
blink_detected = False
blink_start_time = None
blink_count = 0
DOUBLE_BLINK_TIME_WINDOW = 0.5  # Time window for double blink in seconds

# Function to map eye position to mouse position
def map_to_screen(x, y, frame_width, frame_height):
    global previous_eye_x, previous_eye_y

    # Calculate average eye position for smoothing
    eye_positions_x.append(x)
    eye_positions_y.append(y)
    
    if len(eye_positions_x) > SMOOTHING_WINDOW:
        eye_positions_x.pop(0)
        eye_positions_y.pop(0)
    
    smoothed_x = sum(eye_positions_x) // len(eye_positions_x)
    smoothed_y = sum(eye_positions_y) // len(eye_positions_y)
    
    # Only move the cursor if the movement is significant
    if previous_eye_x is None or previous_eye_y is None or \
            abs(smoothed_x - previous_eye_x) > MOVEMENT_THRESHOLD or \
            abs(smoothed_y - previous_eye_y) > MOVEMENT_THRESHOLD:
        
        # Map eye position to screen coordinates, based on the actual frame size
        screen_x = int(smoothed_x * (SCREEN_WIDTH / frame_width))
        screen_y = int(smoothed_y * (SCREEN_HEIGHT / frame_height))

        pyautogui.moveTo(screen_x, screen_y)

        previous_eye_x, previous_eye_y = smoothed_x, smoothed_y

# Function to detect double blinks and trigger mouse click
def detect_double_blink(eyes_detected):
    global blink_detected, blink_start_time, blink_count

    if eyes_detected == 0 and not blink_detected:
        # Eyes are not detected, start blink timer
        current_time = time.time()
        blink_detected = True

        # Check if this blink is within the double blink time window
        if blink_start_time is not None and (current_time - blink_start_time) <= DOUBLE_BLINK_TIME_WINDOW:
            blink_count += 1
        else:
            blink_count = 1  # Reset blink count if too much time has passed

        blink_start_time = current_time

    elif eyes_detected > 0 and blink_detected:
        # Eyes are detected again, reset blink detection
        blink_detected = False

        # Check if double blink occurred
        if blink_count == 2:
            # Trigger a mouse click
            pyautogui.click()
            blink_count = 0  # Reset after a successful double blink

# Function to detect significant head movement and trigger mouse actions
def detect_head_movement(face_center_x, face_center_y):
    global previous_face_x, previous_face_y

    # If this is the first frame, set the previous face position
    if previous_face_x is None or previous_face_y is None:
        previous_face_x, previous_face_y = face_center_x, face_center_y

    # Calculate the difference in face position
    dx = face_center_x - previous_face_x
    dy = face_center_y - previous_face_y

    # Detect significant horizontal movement (left/right)
    if abs(dx) > HEAD_MOVEMENT_THRESHOLD:
        if dx > 0:
            print("Head moved right. Trigger right-click.")
            pyautogui.rightClick()
        else:
            print("Head moved left. Trigger left-click.")
            pyautogui.click()

    # Detect significant vertical movement (up/down)
    if abs(dy) > HEAD_MOVEMENT_THRESHOLD:
        if dy > 0:
            print("Head moved down. Scrolling down.")
            pyautogui.scroll(-1)
        else:
            print("Head moved up. Scrolling up.")
            pyautogui.scroll(1)

    # Update the previous face position
    previous_face_x, previous_face_y = face_center_x, face_center_y

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Set the desired frame width and height (increase resolution)
frame_width = 1280
frame_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Loop to continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally to remove the mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Track the number of eyes detected in this frame
    total_eyes_detected = 0

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Calculate the center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # Detect head movement and trigger actions
        detect_head_movement(face_center_x, face_center_y)

        # Extract the region of interest (ROI) for face detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Update total eyes detected
        total_eyes_detected += len(eyes)

        # Iterate over detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Calculate the center of the eye relative to the frame
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2

            # Map eye position to screen coordinates
            map_to_screen(eye_center_x, eye_center_y, frame_width, frame_height)

    # Detect double blink and trigger a mouse click if needed
    detect_double_blink(total_eyes_detected)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Check for key press; press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
