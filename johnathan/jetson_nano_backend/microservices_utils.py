import sqlite3

def get_latest_imu_data():
    """
    Retrieves the most recent IMU reading from the database.
    Returns:
        float: The most recent IMU value
    Raises:
        sqlite3.Error: If there is a database error 
        ValueError: If no IMU data exists in database
        Exception: If any other error occurs
    """
    try:
        conn = sqlite3.connect('/home/jetson/gps_data.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT value
            FROM imu_data 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            raise ValueError("No IMU data found in database")
            
        return result[0]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

def get_latest_gps_coordinates() -> tuple[float, float]:
    try:
        conn = sqlite3.connect('/home/jetson/gps_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT latitude, longitude FROM gps_coordinates ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            raise ValueError("No GPS coordinates found in database")
            
        return result
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
