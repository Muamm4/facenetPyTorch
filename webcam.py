from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image, ImageDraw
import torch

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a frame from the webcam
ret, frame = cap.read()

# Release the webcam
cap.release()

if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

# Convert the frame to a PIL image
img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Detect faces in the image
boxes, _ = mtcnn.detect(img)

# If faces are detected, draw bounding boxes and extract embeddings
if boxes is not None:
    draw = ImageDraw.Draw(img)
    for box in boxes:
        # Draw bounding boxes on the image
        draw.rectangle(box.tolist(), outline='red', width=3)

    # Convert the image to a tensor
    img_cropped = mtcnn(img)
    if img_cropped is not None:
        # Calculate embeddings for the detected faces
        embeddings = resnet(img_cropped)
        print("Embeddings calculated for detected faces.")
    else:
        print("No faces were cropped for embedding calculation.")
else:
    print("No faces detected.")

# Convert the PIL image back to an OpenCV image and display it
cv2.imshow("Detected Faces", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()