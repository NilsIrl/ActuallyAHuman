import serial
import sqlite3

# Define the fixed database path
DB_PATH = '/home/jetson/gps_data.db'
SERIAL_PORT = '/dev/ttyCH341USB0'  # Adjust this to match your IMU's serial port
BAUD_RATE = 9600  # Changed from 115200 to 9600

def init_database():
    """Initialize the SQLite database with IMU data table"""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # Auto-commit mode
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute('PRAGMA journal_mode=WAL')
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS imu_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.close()

def process_line(line, cursor):
    """
    Process each line of output from the IMU serial connection
    Args:
        line (str): A line of output from the IMU
        cursor: SQLite cursor for database operations
    """
    try:
        # Convert the line to a float
        imu_value = float(line.strip())
        
        # Insert the value into the database
        cursor.execute('INSERT INTO imu_data (value) VALUES (?)', (imu_value,))
        
        # Print for debugging
        print(f"Recorded IMU value: {imu_value}")
    except ValueError as e:
        print(f"Invalid data received: {line.strip()}")

def main():
    # Initialize the database
    init_database()
    
    # Create a single database connection for the entire session
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=rw', uri=True, isolation_level=None)
    cursor = conn.cursor()
    
    # Enable WAL mode for this connection as well
    cursor.execute('PRAGMA journal_mode=WAL')
    
    try:
        # Initialize serial connection
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1
        )
        
        print(f"Connected to IMU on {SERIAL_PORT}")
        
        # Read the serial data line by line
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8')
                process_line(line, cursor)
                
    except KeyboardInterrupt:
        print("\nStopping IMU data collection...")
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
        conn.close()

if __name__ == "__main__":
    main()
