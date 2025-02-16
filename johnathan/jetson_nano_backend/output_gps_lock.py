import subprocess
import shlex
import sqlite3
import re

# Define the fixed database path
DB_PATH = '/home/jetson/gps_data.db'

def init_database():
    """Initialize the SQLite database with GPS coordinates table"""
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # Auto-commit mode
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute('PRAGMA journal_mode=WAL')
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gps_coordinates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL
        )
    ''')
    
    conn.close()

def process_line(line, cursor):
    """
    Process each line of output from the GNSS streamer
    Args:
        line (str): A line of output from the GNSS streamer
        cursor: SQLite cursor for database operations
    """
    # Remove any trailing whitespace and decode from bytes if necessary
    if isinstance(line, bytes):
        line = line.decode('utf-8').strip()
    else:
        line = line.strip()

    # TODO: Add your parsing logic here to extract latitude and longitude
    # This is a placeholder - you'll need to adjust based on your GNSS output format
    # Example: lat, lon = parse_gps_data(line)
    # cursor.execute('INSERT INTO gps_coordinates (latitude, longitude) VALUES (?, ?)', 
    #               (lat, lon))
    
    # For now, just print the line
    # print(f"Received line: {line}")
    # if "lat" in line:
        # print(line)
    BASE_LATITUDE = 37.4298576667
    BASE_LONGITUDE = -122.1720325
    if line.startswith("<NMEA(GNGGA"): # or line.startswith("<NMEA(GNGLL") or line.startswith("<NMEA(GNRMC"):
        # print(line)
        # Extract latitude and longitude using regex
        lat_match = re.search(r'lat=([\d.-]+)', line)
        lon_match = re.search(r'lon=([\d.-]+)', line)
        
        if lat_match and lon_match:
            latitude = float(lat_match.group(1))
            longitude = float(lon_match.group(1))
            
            cursor.execute('INSERT INTO gps_coordinates (latitude, longitude) VALUES (?, ?)', 
                          (latitude, longitude))
        else:
            print("Could not extract coordinates from line:", line)

def main():
    # Initialize the database
    init_database()
    
    # Create a single database connection for the entire session
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=rw', uri=True, isolation_level=None)
    cursor = conn.cursor()
    
    # Enable WAL mode for this connection as well
    cursor.execute('PRAGMA journal_mode=WAL')
    
    # Command to run
    command = [
        "gnssstreamer",
        "-P", "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_GNSS_receiver-if00",
        "--input", "http://rtk2go.com:2101/oakland",
        "--rtkuser", "nils.andre.chang@gmail.com",
        "--rtkpassword", "none"
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Read the output line by line
        while True:
            line = process.stdout.readline()
            if not line:
                break
            process_line(line, cursor)
            
    except KeyboardInterrupt:
        print("\nStopping GNSS streamer...")
        process.terminate()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
        process.wait()

if __name__ == "__main__":
    main()
