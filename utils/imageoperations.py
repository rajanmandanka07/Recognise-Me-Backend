import cv2
import numpy as np
import base64
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def decode_base64_image(base64_string):
    image_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def get_face_embedding_mediapipe(image):
    if image is None:
        return None

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    # Get the first face's landmarks (468 points)
    face_landmarks = results.multi_face_landmarks[0]

    # Create an embedding by flattening the (x, y, z) coordinates of the 468 landmarks
    embedding = []
    for landmark in face_landmarks.landmark:
        embedding.append([landmark.x, landmark.y, landmark.z])

    embedding = np.array(embedding).flatten()
    return embedding

def detect_and_crop_faces(image):
    # Convert image to grayscale for better face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's pre-trained face detector (Haar cascades or DNN-based)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    cropped_faces = []
    for (x, y, w, h) in faces:
        # Crop the face from the image
        face = image[y:y + h, x:x + w]
        cropped_faces.append(face)

    return cropped_faces

def convert_to_grayscale(image):
    # Convert the cropped face image to grayscale (black and white)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
