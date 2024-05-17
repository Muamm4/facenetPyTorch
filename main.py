from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Load an image containing faces
img = Image.open('./fotos/Humans/1 (6982).jpg')

# Detect faces in the image
boxes, _ = mtcnn.detect(img)

print(boxes)
# If faces are detected, extract embeddings
if boxes is not None:
    for box in boxes:
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(box.tolist(), outline='green', width=3)

# Display or save the image with detected faces
img.show()