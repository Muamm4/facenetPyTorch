from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import os
# Initialize MTCNN for face detection
mtcnn = MTCNN()
# Load pre-trained Inception ResNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Load image to check
img_check = Image.open('./natalia1.png')
face_to_check, _ = mtcnn.detect(img_check)
aligned1_to_check = mtcnn(img_check.crop(face_to_check[0].tolist()))
embeddings1_to_check = resnet(aligned1_to_check.unsqueeze(0)).detach()

for box in face_to_check:
            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(img_check)
            draw.rectangle(box.tolist(), outline='green', width=3)   

plt.figure(figsize=(16, 8))    
plt.subplot(2, 5, 3)
plt.imshow(img_check)
files_to_search = os.walk('./fotos')

def scandir_fotos (files_to_search,finder_count,counter):
    for (dirpath, dirnames, filenames) in files_to_search:
        for foto_finder in filenames :
            img_to_compare = Image.open(f'{dirpath}/' + str(foto_finder))
            print(f'{dirpath}/' + str(foto_finder))
            try:
                face_compare, _ = mtcnn.detect(img_to_compare)
                if face_to_check is not None and face_compare is not None:
                    for box in face_compare:
                        try:
                            aligned2_finder = mtcnn(img_to_compare.crop(box.tolist()))
                            embeddings2_finder = resnet(aligned2_finder.unsqueeze(0)).detach()
                            # Calculate the Euclidean distance between embeddings
                            distance = (embeddings1_to_check - embeddings2_finder).norm().item()
                            if distance < 1.0:  # You can adjust the threshold for verification
                                finder_count += 1;
                                same_person = "Same person"
                                # Draw bounding boxes on the image
                                draw = ImageDraw.Draw(img_to_compare)
                                draw.rectangle(box.tolist(), outline='red', width=3)
                                plt.subplot(2, 5, counter)
                                plt.imshow(img_to_compare)
                                counter += 1
                        except:
                            pass  
            except:
                pass   
             
    if finder_count == 0:
        print("Not found")
    else:
        print(f"Found {finder_count} fotos") 
    plt.show()



scandir_fotos(files_to_search,0,6)