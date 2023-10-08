import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Provide the path to the video file
video_path = 'beta_vid.mp4'

# Start capturing video
cap = cv2.VideoCapture(video_path)

# Initialize pose estimation model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define landmarks of interest with human-readable names
landmarks_of_interest = {
    15: 'left_wrist',
    16: 'right_wrist',
    27: 'left_ankle',
    28: 'right_ankle'
}

# List to store all the landmarks for each frame
landmarks_sequence = []

# Desired display width
DISPLAY_WIDTH = 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the pose estimation results
    results = pose.process(rgb_frame)

    # Draw the pose annotations on the frame
    annotated_frame = frame.copy()
    if results.pose_landmarks:  
        frame_landmarks = []
        for idx, name in landmarks_of_interest.items():
            landmark = results.pose_landmarks.landmark[idx]
            frame_landmarks.append({
                'name': name,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
            
            # Draw specific landmarks on the frame
            landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
            if landmark_px: 
                cv2.circle(annotated_frame, landmark_px, 5, (0, 255, 0), -1)
        landmarks_sequence.append(frame_landmarks)

    # Resize the frame for display
    height, width, _ = annotated_frame.shape
    new_height = int((DISPLAY_WIDTH / width) * height)
    resized_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))
    
    # Display the frame
    cv2.imshow('MediaPipe Pose', resized_frame)
    
    # Stop the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Output specific landmarks sequence to a JSON file
with open("specific_landmarks_sequence.json", "w") as file:
    json.dump(landmarks_sequence, file)

# Clean up
cap.release()
cv2.destroyAllWindows()
