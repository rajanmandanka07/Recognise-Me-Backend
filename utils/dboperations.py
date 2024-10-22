import mysql.connector
import os
import pymysql 
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

def get_user_by_id(user_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        query = "SELECT id, full_name, email FROM users WHERE id = %s"
        cursor.execute(query, (user_id,))
        user_data = cursor.fetchone()

        cursor.close()
        connection.close()
        return user_data
    except Exception as e:
        print(f"Error fetching user by ID: {str(e)}")
        return None

# Fetch user's attendance records
def get_user_attendance(user_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        query = "SELECT attendance_date, status FROM attendance WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        attendance_data = cursor.fetchall()

        cursor.close()
        connection.close()
        return attendance_data
    except Exception as e:
        print(f"Error fetching attendance: {str(e)}")
        return None

# Fetch organization details for a user by user ID
def get_organization_by_user_id(user_id):
    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT o.name
            FROM organizations o
            JOIN users u ON u.organization_id = o.id
            WHERE u.id = %s
        """
        cursor.execute(query, (user_id,))
        organization_data = cursor.fetchone()

        cursor.close()
        connection.close()
        return organization_data
    except Exception as e:
        print(f"Error fetching organization: {str(e)}")
        return None

def get_all_attendance_and_user_data():
    query = '''
    SELECT 
        users.full_name, 
        users.email, 
        organizations.name AS organization_name,  -- Fetch the organization name
        attendance.attendance_date, 
        attendance.status 
    FROM 
        attendance 
    JOIN 
        users 
    ON 
        attendance.user_id = users.id
    JOIN 
        organizations  -- Join with organizations table
    ON 
        users.organization_id = organizations.id
    '''
    
    connection = create_connection()  # Create a connection
    if connection is None:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)  # Use dictionary cursor
        cursor.execute(query)
        records = cursor.fetchall()
        return records
    except Exception as e:
        print(f"Error fetching attendance data: {e}")
        return []
    finally:
        if connection.is_connected():
            connection.close()

def get_user_by_email_and_password(email, password):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

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

def mark_attendance(user_id, status):
    """
    Mark the attendance for a given user with the current date and time.
    If the user already has an attendance record for the day, return a message.
    """
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)
    current_datetime = datetime.now()

    try:
        # Insert the attendance record into the attendance table
        cursor.execute(
            "INSERT INTO attendance (user_id, attendance_date, status) VALUES (%s, %s, %s)",
            (user_id, current_datetime, status)
        )
        connection.commit()

        return f"Attendance marked for user {user_id} at {current_datetime}"

    except mysql.connector.errors.IntegrityError as e:
        # Catch the unique constraint violation error
        if "unique_user_attendance_per_day" in str(e):
            return f"Attendance for user {user_id} is already marked for today."
        else:
            return f"An error occurred: {e}"

    finally:
        cursor.close()
        connection.close()
