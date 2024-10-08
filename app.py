import json
from datetime import datetime

from sklearn.cluster import DBSCAN

from flask import Flask, request, jsonify
import mediapipe as mp
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from flask_cors import CORS

# User Registration Endpoint
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json
from scipy.spatial.distance import euclidean


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv('SECRET_KEY')  # Set a secret key for sessions

# Database connection configuration
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Function to create a database connection
def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
    except Error as e:
        print(f"Error: '{e}'")

    return connection

# Define the models for your database tables
def get_user_by_email_and_password(email, password):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

# Handle to get User by ID
def get_attendance_by_user_id(user_id):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM attendance WHERE user_id = %s", (user_id,))
    attendance_records = cursor.fetchall()
    cursor.close()
    connection.close()
    return attendance_records

def get_admin_by_email_and_password(email, password):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM admins WHERE email = %s AND password = %s", (email, password))
    admin = cursor.fetchone()
    cursor.close()
    connection.close()
    return admin

def get_user_by_email(email):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

def create_user(full_name, email, password, organization_id, admin_id):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute(
        "INSERT INTO users (full_name, email, password, organization_id, admin_id) VALUES (%s, %s, %s, %s, %s)",
        (full_name, email, password, organization_id, admin_id)
    )
    connection.commit()
    new_user_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return {"id": new_user_id, "full_name": full_name}

# Function to store face embedding in 'face_data' table
def store_face_embedding(user_id, embedding):
    connection = create_connection()
    cursor = connection.cursor()

    # Convert the embedding array to JSON
    embedding_json = json.dumps(embedding.tolist())

    cursor.execute("INSERT INTO face_data (user_id, embedding) VALUES (%s, %s)", (user_id, embedding_json))
    connection.commit()
    cursor.close()
    connection.close()

# Function to get all attendance records
def get_all_attendance_records():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT users.full_name, attendance.attendance_date, attendance.status 
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        ORDER BY attendance.attendance_date DESC
    """)
    attendance_records = cursor.fetchall()
    cursor.close()
    connection.close()
    return attendance_records

def mark_attendance(user_id, status):
    """
    Mark the attendance for a given user with the current date and time.
    """
    connection = create_connection()
    cursor = connection.cursor()

    # Get the current date and time
    current_datetime = datetime.now()

    # Insert the attendance record into the attendance table
    cursor.execute(
        "INSERT INTO attendance (user_id, attendance_date, status) VALUES (%s, %s, %s)",
        (user_id, current_datetime, status)
    )
    connection.commit()
    cursor.close()
    connection.close()

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

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_base64_image_1(base64_string):
    # Decode the base64 image into a numpy array
    image_data = base64.b64decode(base64_string)
    image_np = np.array(Image.open(io.BytesIO(image_data)))
    return image_np


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

# -- End Points --
# Endpoint for user login
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    # Validate input
    if 'email' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Fetch user data
    user = get_user_by_email_and_password(data['email'], data['password'])

    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401

    # Retrieve attendance data for the user
    attendance_records = get_attendance_by_user_id(user['id'])

    # Use the correct column name 'attendance_date'
    attendance_data = [
        {"date": record['attendance_date'].strftime("%Y-%m-%d"), "status": record['status']}
        for record in attendance_records
    ]

    return jsonify({"user_id": user['id'], "full_name": user['full_name'], "attendance": attendance_data}), 200

# Admin Login Endpoint
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()

    # Validate input
    if 'adminId' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Fetch admin data
    admin = get_admin_by_email_and_password(data['adminId'], data['password'])
    # print(admin)

    if admin is None:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"admin_id": admin['id'], "name": admin['name']}), 200

@app.route('/api/register_user', methods=['POST'])
def register_user():
    data = request.get_json()

    # Validate input fields
    required_fields = ['full_name', 'email', 'password', 'organization_id', 'admin_id', 'images']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"'{field}' is required"}), 400

    full_name = data['full_name']
    email = data['email']
    password = data['password']
    organization_id = data['organization_id']
    admin_id = data['admin_id']
    images_base64 = data['images']  # This should be an array of base64-encoded images

    # Check if the user already exists
    if get_user_by_email(email):
        return jsonify({"error": "User with this email already exists"}), 409

    embeddings = []

    # Process each image to extract embeddings
    try:
        for image_base64 in images_base64:
            image = decode_base64_image(image_base64)

            # Detect faces and crop them
            cropped_faces = detect_and_crop_faces(image)

            if not cropped_faces:
                return jsonify({"error": "No faces detected in one of the images"}), 400

            for face in cropped_faces:
                # Convert to grayscale (black and white)
                gray_face = convert_to_grayscale(face)

                # Get the face embedding
                embedding = get_face_embedding_mediapipe(gray_face)

                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    return jsonify({"error": "Could not generate embedding for one of the images"}), 400

    except Exception as e:
        return jsonify({"error": "Error processing images: " + str(e)}), 400

    # Store user information in the 'users' table
    user = create_user(full_name, email, password, organization_id, admin_id)

    # Store each face embedding in the 'face_data' table
    for embedding in embeddings:
        store_face_embedding(user['id'], embedding)

    return jsonify({"message": "User registered successfully", "user_id": user['id']}), 201

# Admin endpoint to view all students' attendance
@app.route('/api/admin/attendance', methods=['GET'])
def get_all_attendance():
    # You can add admin authorization here if needed

    attendance_records = get_all_attendance_records()

    # Format the attendance data for the response
    formatted_attendance = [
        {
            "full_name": record['full_name'],
            "date": record['attendance_date'].strftime("%Y-%m-%d"),
            "status": record['status']
        }
        for record in attendance_records
    ]

    return jsonify({"attendance": formatted_attendance}), 200

@app.route('/api/compare_faces', methods=['POST'])
def compare_faces():
    data = request.get_json()

    # Check for valid input
    if 'image' not in data:
        return jsonify({"error": "Image is required"}), 400

    # Decode the image from base64
    try:
        image = decode_base64_image(data['image'])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Detect and crop the face
    cropped_faces = detect_and_crop_faces(image)
    if not cropped_faces:
        return jsonify({"error": "No face detected in the provided image"}), 400

    # Process each cropped face
    for face in cropped_faces:
        # Convert the cropped face to grayscale (black and white)
        gray_face = convert_to_grayscale(face)

        # Generate embedding for the provided image
        embedding_input = get_face_embedding_mediapipe(gray_face)

        if embedding_input is None:
            return jsonify({"error": "Could not generate embedding for the provided image."}), 400

        # Retrieve all face embeddings from the 'face_data' table
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT user_id, embedding FROM face_data")
        face_data_records = cursor.fetchall()

        # Convert input embedding to numpy array for comparison
        embedding_input = np.array(embedding_input)

        # Set a threshold for Euclidean distance to consider two faces similar
        threshold = 0.5
        matching_user_id = None
        min_distance = float('inf')

        # Compare input embedding with each stored embedding
        for record in face_data_records:
            user_id = record['user_id']
            embedding_stored = np.array(json.loads(record['embedding']))

            # Calculate the Euclidean distance
            euclidean_distance = euclidean(embedding_input, embedding_stored)

            # If the distance is less than the threshold and is the smallest so far, we have a match
            if euclidean_distance < threshold and euclidean_distance < min_distance:
                matching_user_id = user_id
                min_distance = euclidean_distance

        if matching_user_id is not None:
            # Mark attendance for the matched user
            mark_attendance(matching_user_id, "Present")

            # Retrieve user data (name) by user_id from the 'user' table
            cursor.execute("SELECT full_name FROM users WHERE id = %s", (matching_user_id,))
            user_data = cursor.fetchone()

            if user_data:
                user_name = user_data['full_name']
            else:
                user_name = 'Unknown'

            cursor.close()
            connection.close()

            return jsonify({
                "user_id": matching_user_id,
                "user_name": user_name,  # Return user name
                "euclidean_distance": min_distance,
                "is_similar": True,
                "message": f"Attendance marked as Present for {user_name}"
            }), 200

    cursor.close()
    connection.close()

    return jsonify({
        "message": "No matching face found",
        "is_similar": False
    }), 404

mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
@app.route('/upload', methods=['POST'])
def process_photos():
    data = request.get_json()
    base64_images = data['images']  # List of base64-encoded images

    face_encodings = []
    face_photos = []

    # Process the base64 images
    for base64_img in base64_images:
        image = base64_to_image(base64_img)

        # Convert to RGB as MediaPipe expects RGB input
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face landmarks using MediaPipe FaceMesh
        results = mp_face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])

                # Flatten the landmarks into a 1D array to use as "embeddings"
                face_encodings.append(landmarks.flatten())

                # Convert the image back to base64 to return it
                face_photos.append(image_to_base64(image))

    # Perform clustering based on the embeddings (landmarks)
    if len(face_encodings) > 0:
        face_encodings = np.array(face_encodings)
        clustering = DBSCAN(metric='euclidean', eps=1.0, min_samples=1).fit(face_encodings)

        # Group photos by their cluster labels
        groups = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in groups:
                groups[label] = []
            groups[label].append(face_photos[idx])  # Store the base64-encoded image

        # Prepare response
        response = {'groups': [{'photos': group} for group in groups.values()]}
        return jsonify(response)
    else:
        return jsonify({'groups': []})


if __name__ == '__main__':
    app.run(debug=True)
