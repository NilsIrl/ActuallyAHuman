import sqlite3
import time
import serial
from roboclaw import Roboclaw
def get_latest_imu_data(conn_obj = None):
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
        if conn_obj is None:
            conn = sqlite3.connect('/home/jetson/gps_data.db')
        else:
            conn = conn_obj
        cursor = conn.cursor()
        cursor.execute('''
            SELECT value
            FROM imu_data 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        result = cursor.fetchone()
        if conn_obj is None:
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
    conn = sqlite3.connect('/home/jetson/gps_data.db')
    current_heading = get_latest_imu_data(conn)
    desired_heading = current_heading + rotation_degrees
    
    # Use monotonic time for reliable timing with fixed 5 second timeout
    start_time = time.monotonic()
    
    TOLERANCE = 5  # Hardcoded tolerance value in degrees
    
    while time.monotonic() - start_time < 5.0:
        # Get current heading and calculate error
        current_heading = get_latest_imu_data(conn)
        angle_difference = desired_heading - current_heading
        print(f"desired_heading: {desired_heading}, current_heading: {current_heading}")
        print(f"angle_difference: {angle_difference}")
        # Check if we've reached the target within tolerance
        # if abs(angle_difference) < TOLERANCE:
        #     # Stop motors and return success
        #     rc.ForwardBackwardM1(address, 64)
        #     rc.ForwardBackwardM2(address, 64)
        #     return True
            
        P_GAIN = 0.7
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
    return True

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

def connect_to_arduino():
    """
    Establishes a serial connection to the Arduino.
    
    Returns:
        serial.Serial: Serial connection object to the Arduino
        
    Raises:
        SerialException: If connection to Arduino fails
    """
    ARDUINO_PORT = '/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_1423831383435181F1B0-if00'  # Standard Arduino USB port on Linux
    BAUD_RATE = 9600
    
    return serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

def _control_servo(arduino_serial, servo_type: int, angle: int) -> None:
    """
    Controls the servo motors connected to the Arduino.
    
    Args:
        arduino_serial: Serial connection to Arduino
        servo_type: 1 for pan servo, 2 for extend servo
        angle: Servo angle (0-180 degrees)
        
    Raises:
        ValueError: If invalid servo_type or angle provided
        SerialException: If communication with Arduino fails
    """
    if servo_type not in [1, 2]:
        raise ValueError("servo_type must be 1 (pan) or 2 (extend)")
    if not 0 <= angle <= 180:
        raise ValueError("angle must be between 0 and 180 degrees")
    
    try:
        # Send servo selector (1 or 2)
        arduino_serial.write(bytes([servo_type]))
        # Send angle value
        arduino_serial.write(bytes([angle]))
    except serial.SerialException as e:
        raise serial.SerialException(f"Failed to communicate with Arduino: {e}")

def pan_servo(arduino_serial, angle: int) -> None:
    """
    Controls the pan servo motor.
    Angle 0 maps to center position (90 degrees).
    Other angles are scaled by 1.5 and offset from center.
    
    Args:
        arduino_serial: Serial connection to Arduino
        angle: Pan angle, where 0 is center position
    """
    _control_servo(arduino_serial, 1, int((angle / 1.5) + 90))

def extend(arduino_serial) -> None:
    """
    Extends the servo motor to the "out" position (60 degrees).
    
    Args:
        arduino_serial: Serial connection to Arduino
    """
    _control_servo(arduino_serial, 2, 60)

def retract(arduino_serial) -> None:
    """
    Retracts the servo motor to the "in" position (100 degrees).
    
    Args:
        arduino_serial: Serial connection to Arduino
    """
    _control_servo(arduino_serial, 2, 100)


def initialize_arm():
    """
    Initializes the arm's roboclaw controller with the necessary settings.
    
    Args:
        rc: Roboclaw instance
        address: Roboclaw address (default: 0x80)
    """
    rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.2:1.0", 115200)
    rc.Open()
    address = 0x80
    # Reset encoders to 0 (assuming arm is at bottom position)
    rc.ResetEncoders(address)
    
    # Configure PID settings
    rc.SetM2VelocityPID(address, 0, 0, 0, 2500)
    rc.SetM2PositionPID(address, 200, 0, 4000, 100, 10, 0, 10500)

    # Ensure motor is stopped
    rc.ForwardM2(address, 0)
    return rc

def set_arm_position(rc, position: int) -> None:
    """
    Sets the arm to a specific position using encoder counts.
    
    Args:
        rc: Roboclaw instance
        position: Desired position in encoder counts (0-10500)

    Raises:
        ValueError: If position is outside valid range
    """
    rc.SetM2Position(0x80, position, 1)
