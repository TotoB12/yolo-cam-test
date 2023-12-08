from ultralytics import YOLO
import cv2
# Import emotion recognition library (example: deepface)
from deepface import DeepFace

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the camera source
source = '0'

# Stream results from the camera
results = model(source, stream=True)

for result in results:
    for box in result.boxes.data:
        if model.names[int(box[5])] == 'person':
            xmin, ymin, xmax, ymax = map(int, box[:4])
            face_img = result.orig_img[ymin:ymax, xmin:xmax]

            try:
                # Perform emotion analysis
                analysis = DeepFace.analyze(face_img, actions=['emotion'])
                
                # Access the dominant emotion from the first element of the analysis list
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_label = dominant_emotion
            except Exception as e:
                print(f"Error in emotion detection: {str(e)}")  # Print the specific error
                emotion_label = "Error in emotion detection"

            # Overlay the emotion label on the video stream
            cv2.putText(result.orig_img, emotion_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detections and emotions
    cv2.imshow('YOLOv8 Detection and Emotion Recognition', result.orig_img)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
