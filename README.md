# Recognise Me Backend

This is the backend for the "Recognise Me" project, a facial recognition-based attendance system. The backend is built using Flask, MySQL, and Mediapipe for face embeddings. It handles user authentication, face recognition, and attendance marking.

## Table of Contents

- [Features](#features)
- [API Endpoints](#api-endpoints)
- [Key Functions](#key-functions)
- [Error Handling](#error-handling)
- [Security](#security)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

## Features

- **User Registration**: Users can register with their details and provide images for facial recognition.
- **User Login**: Users can log in with their credentials and view their attendance history.
- **Admin Login**: Admins can log in and view all users' attendance.
- **Attendance Marking**: Automatically mark attendance based on facial recognition.
- **Face Comparison**: The backend compares captured face embeddings with stored embeddings to identify users.

## API Endpoints

### 1. User Login
- **URL**: `/api/login`
- **Method**: `POST`
- **Description**: Allows users to log in using their email and password.
- **Request Body**:
    ```json
    {
      "email": "user@example.com",
      "password": "user_password"
    }
    ```
- **Response**:
    ```json
    {
      "user_id": 1,
      "full_name": "John Doe",
      "attendance": [
        {
          "date": "2024-10-01",
          "status": "Present"
        }
      ]
    }
    ```

### 2. Admin Login
- **URL**: `/api/admin/login`
- **Method**: `POST`
- **Description**: Admin can log in using email and password.
- **Request Body**:
    ```json
    {
      "adminId": "admin@example.com",
      "password": "admin_password"
    }
    ```
- **Response**:
    ```json
    {
      "admin_id": 1,
      "name": "Admin Name"
    }
    ```

### 3. Register User
- **URL**: `/api/register_user`
- **Method**: `POST`
- **Description**: Registers a new user with details and face embeddings.
- **Request Body**:
    ```json
    {
      "full_name": "John Doe",
      "email": "john@example.com",
      "password": "user_password",
      "organization_id": 1,
      "admin_id": 1,
      "images": [
        "base64_image_string"
      ]
    }
    ```
- **Response**:
    ```json
    {
      "message": "User registered successfully",
      "user_id": 1
    }
    ```

### 4. Admin View Attendance
- **URL**: `/api/admin/attendance`
- **Method**: `GET`
- **Description**: Fetches all users' attendance records.
- **Response**:
    ```json
    {
      "attendance": [
        {
          "full_name": "John Doe",
          "date": "2024-10-01",
          "status": "Present"
        }
      ]
    }
    ```

### 5. Compare Faces for Attendance
- **URL**: `/api/compare_faces`
- **Method**: `POST`
- **Description**: Compares a provided face image with stored embeddings to recognize a user and mark attendance.
- **Request Body**:
    ```json
    {
      "image": "base64_image_string"
    }
    ```
- **Response**:
    ```json
    {
      "message": "Attendance marked for John Doe"
    }
    ```

## Key Functions

### Database Interaction
- **create_connection()**: Establishes a connection to the MySQL database.
- **get_user_by_email_and_password()**: Fetches user data by email and password.
- **get_attendance_by_user_id()**: Retrieves a user's attendance records.
- **get_admin_by_email_and_password()**: Fetches admin data by email and password.
- **store_face_embedding()**: Stores face embeddings in the `face_data` table.
- **mark_attendance()**: Marks attendance for a user.

### Face Processing
- **decode_base64_image()**: Decodes a base64 string into an image.
- **get_face_embedding_mediapipe()**: Extracts face embeddings using Mediapipe's FaceMesh.
- **detect_and_crop_faces()**: Detects and crops faces from an image using OpenCV.
- **convert_to_grayscale()**: Converts an image to grayscale.

## Error Handling
- The backend includes basic error handling for invalid input, such as missing fields or invalid images.
- **Error Codes**: 
    - 400: Bad Request (Invalid Input)
    - 401: Unauthorized (Invalid Credentials)
    - 409: Conflict (User Already Exists)

## Security
- **Secret Key**: A secret key for sessions is loaded from environment variables.
- **CORS**: CORS is enabled to allow communication between the frontend and backend.

## Technologies Used
- **Flask**: Web framework used for API endpoints.
- **MySQL**: Database used for storing user and attendance data.
- **Mediapipe**: Library used for facial recognition and embedding extraction.
- **OpenCV**: Used for face detection.
- **MySQL Connector**: For MySQL database interaction.

## Contributors
This project is maintained by:

- **Rajankumar Mandanka**
- **Hitest Rabadiya**
- **Rushi Lukka**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.