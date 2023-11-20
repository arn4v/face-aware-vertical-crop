import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def get_bounding_box(pose_landmarks, image_width, image_height):
    landmarks = pose_landmarks.landmark
    x_min = min([landmark.x for landmark in landmarks if landmark.visibility > 0.5], default=0)
    y_min = min([landmark.y for landmark in landmarks if landmark.visibility > 0.5], default=0)
    x_max = max([landmark.x for landmark in landmarks if landmark.visibility > 0.5], default=1)
    y_max = max([landmark.y for landmark in landmarks if landmark.visibility > 0.5], default=1)

    # Convert from relative to absolute coordinates.
    x_min = int(x_min * image_width)
    y_min = int(y_min * image_height)
    x_max = int(x_max * image_width)
    y_max = int(y_max * image_height)

    # Add some padding.
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)

    return (x_min, y_min, x_max - x_min, y_max - y_min)

def detect_people(frame):
    image_height, image_width, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return []
    
    bounding_boxes = [get_bounding_box(results.pose_landmarks, image_width, image_height)]
    
    return bounding_boxes

def crop_frame(frame, box):
    x, y, w, h = box
    return frame[y:y+h, x:x+w]

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties.
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video resolution for 9:16 720p video
    output_width = 720
    output_height = 1280
    
    # Prepare VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect people in the frame.
        boxes = detect_people(frame)
        
        # Create crops based on the number of boxes.
        if len(boxes) == 1:
            # If there's only one person, crop the frame around them and resize to 720p.
            crop = crop_frame(frame, boxes[0])
            crop = cv2.resize(crop, (output_width, output_height))
            out.write(crop)
        elif len(boxes) == 2:
            # If there are two people, create two crops and resize them to fit in 720p.
            crop1 = crop_frame(frame, boxes[0])
            crop2 = crop_frame(frame, boxes[1])
            crop1 = cv2.resize(crop1, (output_width, int(output_height / 2)))
            crop2 = cv2.resize(crop2, (output_width, int(output_height / 2)))
            stacked_crop = cv2.vconcat([crop1, crop2])
            out.write(stacked_crop)
        else:
            # Handle other cases, e.g., no people detected.
            continue  # Skip the frame or handle as needed.
    
    cap.release()
    out.release()


# Call the processing function with the path to your input video and the desired output path.
process_video('zuckerberg_trimmed.mp4', 'zuckerberg_trimmed-cropped.mp4')
