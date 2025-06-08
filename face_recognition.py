import cv2
import matplotlib.pyplot as plt

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Cascade not loaded properly!")

# Load image
img = cv2.imread('sample_face20.jpg')
if img is None:
    print("Image not loaded!")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print("Faces found:", len(faces))

# Make a copy of the image 
output_img = img.copy()

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Faces')
plt.show()
