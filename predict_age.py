from facenet_pytorch import MTCNN, InceptionResnetV1
from age_gender_predictor import AgeGenderPredictor
from PIL import Image, ImageDraw

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load pre-trained Inception ResNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Initialize AgeGenderPredictor
predictor = AgeGenderPredictor()

# Load an image with a face
img = Image.open('path_to_image.jpg')

# Detect faces and extract embeddings
faces, _ = mtcnn.detect(img)

if faces is not None:
    aligned = mtcnn(img)
    embeddings = resnet(aligned).detach()
    
    draw = ImageDraw.Draw(img)
    draw.rectangle(box.tolist(), outline='green', width=3)
    # Predict age and gender
    age, gender = predictor.predict_age_gender(embeddings)
    print(f"Predicted Age: {age} years")
    print(f"Predicted Gender: {gender}")

        