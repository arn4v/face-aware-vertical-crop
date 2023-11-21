import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def get_bounding_box(pose_landmarks, image_width, image_height):
    landmarks = pose_landmarks.landmark
    # Consider only landmarks with a visibility higher than a threshold.
    visible_landmarks = [landmark for landmark in landmarks if landmark.visibility > 0.5]
    if not visible_landmarks:
        # If no landmarks are visible, return an empty box.
        return 0, 0, image_width, image_height

    x_min = min([landmark.x for landmark in visible_landmarks])
    y_min = min([landmark.y for landmark in visible_landmarks])
    x_max = max([landmark.x for landmark in visible_landmarks])
    y_max = max([landmark.y for landmark in visible_landmarks])

    # Convert from relative to absolute coordinates and add some padding.
    padding = 10  # You might need to adjust this padding.
    x_min = int(x_min * image_width) - padding
    y_min = int(y_min * image_height) - padding
    x_max = int(x_max * image_width) + padding
    y_max = int(y_max * image_height) + padding

    # Clamp values to be within frame dimensions.
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    return x_min, y_min, x_max - x_min, y_max - y_min

def detect_people(frame):
    image_height, image_width, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return []
    
    bounding_boxes = [get_bounding_box(results.pose_landmarks, image_width, image_height)]
    
    return bounding_boxes

def resize_and_pad(crop, desired_width, desired_height):
    # Calculate the new size to maintain aspect ratio.
    h, w = crop.shape[:2]
    scale = min(desired_width / w, desired_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_crop = cv2.resize(crop, (new_w, new_h))

    # Create a new canvas with the desired size.
    canvas = cv2.copyMakeBorder(
        resized_crop,
        top=(desired_height - new_h) // 2,
        bottom=(desired_height - new_h) // 2,
        left=(desired_width - new_w) // 2,
        right=(desired_width - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding.
    )

    return canvas

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
        for box in boxes:
            crop = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            canvas = resize_and_pad(crop, output_width, output_height)
            out.write(canvas)
    
    cap.release()
    out.release()

# Call the processing function with the path to your input video and the desired output path.
process_video('zuckerberg_trimmed.mp4', 'zuckerberg_trimmed-cropped.mp4')
