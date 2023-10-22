import json 
import instaloader
import requests
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import re
import os
from requests.exceptions import MissingSchema
from instaloader.exceptions import BadResponseException
import shutil
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Instaloader and login
L = instaloader.Instaloader()
username = "XXX"
password = "XXX"
L.context.login(username, password)

df = pd.read_csv('.\instagram_links.csv')
print(df.head())

id = 5
input_string = df.link_and_username[id]
climb_name = df.name[id]

# This pattern is updated to include parentheses around the shortcode part, which will make it a capturing group
pattern = r'https://www\.instagram\.com/p/([\w-]+)/'

# Find all matches in the input string and extract the shortcode
matches = re.findall(pattern, input_string)
print(matches)
# Print the shortcode
for counter, shortcode in enumerate(matches):
    # Create new directory if it doesn't exist
    output_directory = f'.\\vids\\{id}_{climb_name}\\{counter}'
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode) # replace 'shortcode_here'
    except BadResponseException :
        continue

    video_url = post.video_url

    # Stream video
    try: 
        r = requests.get(video_url, stream=True)
    except MissingSchema:
        print('Invalid URL')
        continue
    if r.status_code == 200:
        byte_stream = bytes()
        with open('temp_video.mp4', 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if os.path.exists(f'{output_directory}\\{shortcode}.mp4'):
        continue
    cap = cv2.VideoCapture("temp_video.mp4")

    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{output_directory}\\{shortcode}.mp4', fourcc, original_fps, (frame_width, frame_height))

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

    # Define colors in HSV
    colors = {
        'Start': ([50, 100, 100], [70, 255, 255]),  # Green
        'Middle': ([80, 100, 100], [100, 255, 255]),  # Cyan
        'Finish': ([130, 100, 100], [150, 255, 255]),  # Magenta
    }

    # Desired display width
    DISPLAY_WIDTH = 400

    # Define a flag to indicate first frame
    is_first_frame = True

    # To store bounding boxes of colored objects in normalized coordinates
    normalized_static_objects = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
                # Convert the BGR image to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if is_first_frame:
            # Detect colored objects only in the first frame
            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                normalized_static_objects[color_name] = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Normalize the bounding box coordinates
                    x_norm = x / frame.shape[1]
                    y_norm = y / frame.shape[0]
                    w_norm = w / frame.shape[1]
                    h_norm = h / frame.shape[0]
                    
                    normalized_static_objects[color_name].append((x_norm, y_norm, w_norm, h_norm))
            is_first_frame = False
        if normalized_static_objects['Start'] == [] or normalized_static_objects['Finish'] == []:
            cap.release()
            out.release()
            shutil.rmtree(output_directory)
            continue

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


        
        # Draw bounding boxes for all detected objects
        for color_name, bboxes in normalized_static_objects.items():
            for (x, y, w, h) in bboxes:
                # Convert normalized coordinates back to pixel for drawing
                x_px, y_px, w_px, h_px = int(x * frame.shape[1]), int(y * frame.shape[0]), int(w * frame.shape[1]), int(h * frame.shape[0])
                cv2.rectangle(annotated_frame, (x_px, y_px), (x_px+w_px, y_px+h_px), (0, 255, 0), 2)

        # Resize the frame for display
        height, width, _ = annotated_frame.shape
        new_height = int((DISPLAY_WIDTH / width) * height)
        resized_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, new_height))
        cv2.imshow('MediaPipe Pose', resized_frame)
        if annotated_frame is not None:
            out.write(annotated_frame)
        # Stop the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    if os.path.exists(output_directory):
        # Output specific landmarks sequence to a JSON file
        with open(f"{output_directory}\\lm_seq.json", "w") as file:
            json.dump(landmarks_sequence, file)

            # Output static objects to a JSON file in normalized coordinates
        with open(f"{output_directory}\\lm_static.json", "w") as file:
            json.dump(normalized_static_objects, file)

    # Clean up
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
