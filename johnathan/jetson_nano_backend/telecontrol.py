import time
import serial  # Change this line
from roboclaw import Roboclaw
from microservices_utils import rotate_robot, move_robot_forward_time, get_latest_imu_data
import keyboard

# Initialize Roboclaw
try:
    rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0", 115200)
    rc.Open()
    address = 0x80

    # Configure PID settings
    rc.SetM1PositionPID(address, 200, 0, 4000, 100, 10, -2_000_000_000, 2_000_000_000)
    rc.SetM2PositionPID(address, 200, 0, 4000, 100, 10, -2_000_000_000, 2_000_000_000)
    rc.SetM1VelocityPID(address, 400, 0, 0, 2500)
    rc.SetM2VelocityPID(address, 400, 0, 0, 2500)
except Exception as e:
    print(f"Error initializing Roboclaw: {e}")
    exit(1)

def print_controls():
    print("\nTeleop Controls:")
    print("W - Move Forward")
    print("A - Turn Left")
    print("D - Turn Right")
    print("Q - Quit")
    print("H - Show current heading")
    print("? - Show this help message")

def main():
    print_controls()
    
    try:
        while True:
            if keyboard.is_pressed('w'):
                print("Moving forward...")
                move_robot_forward_time(rc, 0.5, address)  # Move forward 0.5 meters
                time.sleep(0.1)  # Small delay to prevent repeated triggers
                
            elif keyboard.is_pressed('a'):
                print("Turning left...")
                rotate_robot(rc, -45, address)  # Turn left 45 degrees
                time.sleep(0.1)
                
            elif keyboard.is_pressed('d'):
                print("Turning right...")
                rotate_robot(rc, 45, address)  # Turn right 45 degrees
                time.sleep(0.1)
                
            elif keyboard.is_pressed('h'):
                try:
                    current_heading = get_latest_imu_data()
                    print(f"Current heading: {current_heading:.2f} degrees")
                except Exception as e:
                    print(f"Error getting IMU data: {e}")
                time.sleep(0.1)
                
            elif keyboard.is_pressed('?'):
                print_controls()
                time.sleep(0.1)
                
            elif keyboard.is_pressed('q'):
                print("Quitting teleop control...")
                break
                
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Make sure motors are stopped
        rc.ForwardBackwardM1(address, 64)
        rc.ForwardBackwardM2(address, 64)

if __name__ == "__main__":
    main()