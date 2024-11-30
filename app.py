from datetime import datetime

from flask import Flask, request, jsonify
import mediapipe as mp
import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from flask_cors import CORS
from queue import Queue
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json
from scipy.spatial.distance import euclidean

from utils.dboperations import create_connection, get_organization_by_user_id, get_user_attendance, get_user_by_id, get_all_attendance_and_user_data, get_user_by_email_and_password, get_admin_by_email_and_password, get_user_by_email, create_user, store_face_embedding, mark_attendance
from utils.imageoperations import decode_base64_image, get_face_embedding_mediapipe, detect_and_crop_faces, convert_to_grayscale

from scipy.spatial.distance import cosine
from deepface import DeepFace
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

app = Flask(__name__)
CORS(app)

# Initialize the queue for embeddings
embeddings_queue = Queue()

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

    return jsonify({"user_id": user['id'], "full_name": user['full_name']}), 200

@app.route('/api/user/dashboard', methods=['POST'])
def get_user_dashboard():
    try:
        data = request.get_json()
        user_id = data.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Fetch user details from the database
        user_data = get_user_by_id(user_id)

        if not user_data:
            return jsonify({"error": "User not found"}), 404

        # Fetch user's attendance records
        attendance_data = get_user_attendance(user_id)

        # Fetch organization details for the user
        organization_data = get_organization_by_user_id(user_id)

        # Prepare the response data
        response_data = {
            "full_name": user_data['full_name'],
            "email": user_data['email'],
            "user_id": user_data['id'],
            "attendance": attendance_data,
            "organization": organization_data['name'] if organization_data else "No organization found"
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()

    if 'adminId' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Fetch admin data
    admin = get_admin_by_email_and_password(data['adminId'], data['password'])

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
    images_base64 = data['images']

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

@app.route('/api/admin/attendance', methods=['GET'])
def get_all_attendance():
    # Use the utility function to fetch data
    attendance_records = get_all_attendance_and_user_data()

    # Format the data
    formatted_attendance = [
        {
            "full_name": record['full_name'],
            "email": record['email'],
            "organization_name": record['organization_name'],  # Change to organization_name
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

        # Set a threshold for Euclidean distance to consider two faces
        threshold = 0.5
        matching_user_id = None
        min_distance = float('inf')
        min_similarity = float('inf')

        # Compare input embedding with each stored embedding
        for record in face_data_records:
            user_id = record['user_id']
            embedding_stored = np.array(json.loads(record['embedding']))

            # # Calculate the Euclidean distance
            euclidean_distance = euclidean(embedding_input, embedding_stored)
            print("Euclidean distance : ", euclidean_distance)

            # If the distance is less than the threshold and is the smallest so far, we have a match
            if euclidean_distance < threshold and euclidean_distance < min_distance:
                matching_user_id = user_id
                min_distance = euclidean_distance
            
            # min_similarity = 0.0
            # Calculate the cosine similarity
            # cosine_similarity = cosine(embedding_input, embedding_stored)
            # print("Cosine similarity : ", cosine_similarity)

            # # If the similarity is below the threshold and is the smallest so far, we have a match
            # if cosine_similarity < threshold and cosine_similarity < min_similarity:
            #     matching_user_id = user_id
            #     min_similarity = cosine_similarity

        if matching_user_id is not None:
            # Check and Mark attendance for the matched user and return the result message
            attendance_message = mark_attendance(matching_user_id, "Present")

            # Retrieve user data (name) by user_id from the 'users' table
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
                "user_name": user_name,
                "euclidean_distance": min_distance,
                "is_similar": True,
                "message": f"{attendance_message} for {user_name}"
            }), 200

        cursor.close()
        connection.close()

    return jsonify({
        "message": "No matching face found",
        "is_similar": False
    }), 404

@app.route('/api/mark_group_attendance', methods=['POST'])
def process_and_mark_attendance():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    # Step 1: Read the image from the request
    image_file = request.files['image']
    image_array = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Step 2: Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Step 3: Detect faces
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return jsonify({"success": False, "message": "No faces detected."}), 400

    cropped_faces = []
    base64_faces = []
    h, w, _ = image.shape
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y, width, height = (
            int(bbox.xmin * w),
            int(bbox.ymin * h),
            int(bbox.width * w),
            int(bbox.height * h),
        )
        face = image[y:y + height, x:x + width]
        cropped_faces.append(face)

        # Encode the cropped face as Base64
        _, buffer = cv2.imencode('.jpg', face)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_faces.append(face_base64)

    # Step 4: Generate embeddings and push to the queue
    for face in cropped_faces:
        embedding = get_face_embedding_mediapipe(face)  # Use the new Mediapipe embedding function
        
        if embedding is not None:
            embeddings_queue.put(embedding)
        else:
            print("No face landmarks detected for the current face.")

    # Step 5: Retrieve all embeddings from the database
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT user_id, embedding FROM face_data")
    face_data = cursor.fetchall()
    connection.close()

    user_embedding_map = {}
    for record in face_data:
        user_id = record['user_id']
        embedding = np.array(json.loads(record['embedding']))
        user_embedding_map.setdefault(user_id, []).append(embedding)

    # Print the user_embedding_map for debugging
    # print("User Embedding Map:")
    # for user_id, embeddings in user_embedding_map.items():
    #     print(f"User ID: {user_id}, Embeddings Count: {len(embeddings)}")
    #     for i, embedding in enumerate(embeddings):
    #         print(f"  Embedding {i + 1}: {embedding}")


    # Step 6: Process the queue and find matches
    attendance_user_ids = []
    while not embeddings_queue.empty():
        target_embedding = embeddings_queue.get()
        # target_embedding = normalize_embedding(target_embedding, 128)  # Normalize to 128 dimensions
    
        # Debug: Log target embedding shape
        print(f"Target embedding shape: {len(target_embedding)}")

        for user_id, embeddings in user_embedding_map.items():
            for embedding in embeddings:
                # embedding = normalize_embedding(embedding, 128)  # Normalize to 128 dimensions
                
                # Debug: Log user embedding shape
                # print(f"User ID: {user_id}, Embedding shape: {len(embedding)}")
                
                # Ensure embeddings are the same length
                if len(embedding) != len(target_embedding):
                    print(f"Shape mismatch for User ID {user_id}. Skipping this embedding.")
                    continue
                
                # Calculate Euclidean distance
                euclidean_distance = euclidean(embedding, target_embedding)
                euclidean_distance = euclidean_distance - 1.5
                print(f"User ID: {user_id}, Distance: {euclidean_distance:.4f}")
                
                if euclidean_distance <= 0.5:  # Threshold
                    attendance_user_ids.append(user_id)
                    break


    print(attendance_user_ids)


    # Step 7: Mark attendance for matched users
    marked_users = {}
    for user_id in attendance_user_ids:
        mark_attendance(user_id, "Present")
        user_data = get_user_by_id(user_id)
        if user_data:
            marked_users[user_id] = user_data['full_name']


    # Step 8: Return response with cropped faces and attendance info
    return jsonify({
        "success": True,
        "message": "Attendance marked for detected faces.",
        "attendance": marked_users,
        "cropped_faces": base64_faces,  # Include Base64 encoded cropped faces for testing
    }), 200


if __name__ == '__main__':
    app.run(debug=True)