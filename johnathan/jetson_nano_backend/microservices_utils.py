import sqlite3
import time

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

def rotate_robot(rc, rotation_degrees: float, address: int = 0x80) -> bool:
    """
    Rotates the robot by a specified number of degrees using IMU feedback.
    
    Args:
        rc: Roboclaw instance
        rotation_degrees: Degrees to rotate (positive = clockwise, negative = counter-clockwise)
        address: Roboclaw address (default: 0x80)
    
    Returns:
        bool: True if rotation succeeded within tolerance, False if timed out
        
    Raises:
        Same exceptions as get_latest_imu_data()
    """
    # Get starting heading and calculate target
    current_heading = get_latest_imu_data()
    desired_heading = current_heading + rotation_degrees
    
    # Use monotonic time for reliable timing with fixed 5 second timeout
    start_time = time.monotonic()
    
    TOLERANCE = 2.5  # Hardcoded tolerance value in degrees
    
    while time.monotonic() - start_time < 5.0:
        # Get current heading and calculate error
        current_heading = get_latest_imu_data()
        angle_difference = desired_heading - current_heading
        
        # Check if we've reached the target within tolerance
        if abs(angle_difference) < TOLERANCE:
            # Stop motors and return success
            rc.ForwardBackwardM1(address, 64)
            rc.ForwardBackwardM2(address, 64)
            return True
            
        P_GAIN = 3.0
        # Calculate motor powers using fixed P control (p_gain = 3.0)
        turnpower = angle_difference * P_GAIN
        turnpower_m2 = -turnpower
        
        # Clamp values to valid range
        turnpower = max(-64, min(63, turnpower))
        turnpower_m2 = max(-64, min(63, turnpower_m2))
        
        # Convert to roboclaw format (0-127 range)
        final_power_m1 = int(turnpower + 64)
        final_power_m2 = int(turnpower_m2 + 64)
        
        # Apply powers to motors
        rc.ForwardBackwardM1(address, final_power_m1)
        rc.ForwardBackwardM2(address, final_power_m2)
    
    # If we get here, we timed out - stop motors
    rc.ForwardBackwardM1(address, 64)
    rc.ForwardBackwardM2(address, 64)
    return False

def move_robot_forward_time(rc, distance_meters: float, address: int = 0x80) -> bool:
    """
    Moves the robot forward by a specified distance using time-based control.
    
    Args:
        rc: Roboclaw instance
        distance_meters: Distance to move forward in meters
        address: Roboclaw address (default: 0x80)
    
    Returns:
        bool: True if movement completed, False if there was an error
    """
    # Constants
    SPEED = 2500  # Motor speed (same as in original code)
    FEET_PER_SECOND = 3  # Robot's speed in ft/s
    METERS_PER_SECOND = FEET_PER_SECOND * 0.3048  # Convert to m/s
    
    try:
        # Calculate time needed to move the desired distance
        time_needed = distance_meters / METERS_PER_SECOND
        
        # Start movement
        rc.SpeedM1M2(address, SPEED, SPEED)
        
        # Wait for calculated time
        time.sleep(time_needed)
        
        
    except Exception as e:
        raise Exception(f"Error during movement: {e}") from e
        
    finally:
        # Ensure motors are stopped regardless of any outcome
        rc.SpeedM1M2(address, 0, 0)

