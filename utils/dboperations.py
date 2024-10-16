import mysql.connector
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from mysql.connector import Error 

load_dotenv()

db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}


def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
    except Error as e:
        print(f"Error: '{e}'")

    return connection

def get_user_by_email_and_password(email, password):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

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

def store_face_embedding(user_id, embedding):
    connection = create_connection()
    cursor = connection.cursor()

    # Convert the embedding array to JSON
    embedding_json = json.dumps(embedding.tolist())

    cursor.execute("INSERT INTO face_data (user_id, embedding) VALUES (%s, %s)", (user_id, embedding_json))
    connection.commit()
    cursor.close()
    connection.close()

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
