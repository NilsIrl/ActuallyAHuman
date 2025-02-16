import time
import serial  # Change this line
from roboclaw import Roboclaw
from microservices_utils import rotate_robot, move_robot_forward_time, get_latest_imu_data

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

def get_key():
    import tty
    import sys
    import termios
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()

def main():
    version = rc.ReadVersion(address)
    if not version[0]:
        print("GETVERSION Failed")
    else:
        print(repr(version[1]))
    
    print("\nTeleop Controls:")
    print("w - Move Forward")
    print("a - Turn Left")
    print("d - Turn Right")
    print("q - Quit")
    print("h - Show current heading")
    print("? - Show this help message")
    
    while True:
        key = get_key()
        
        if key == 'q':
            print("Quitting...")
            break
        elif key == 'w':
            print("Moving forward")
            move_robot_forward_time(rc, 0.5, address)
            time.sleep(0.1)
        elif key == 'a':
            print("Turning left")
            rotate_robot(rc, -10, address)  # Turn left 45 degrees
            time.sleep(0.1)
        elif key == 'd':
            print("Turning right")
            rotate_robot(rc, 10, address)  # Turn right 45 degrees
            time.sleep(0.1)
        elif key == 'h':
            print("Current heading: ...")
            try:
                current_heading = get_latest_imu_data()
                print(f"Current heading: {current_heading:.2f} degrees")
            except Exception as e:
                print(f"Error getting IMU data: {e}")
            time.sleep(0.1)
        elif key == '?':
            print("\nTeleop Controls:")
            print("w - Move Forward")
            print("a - Turn Left")
            print("d - Turn Right")
            print("q - Quit")
            print("h - Show current heading")
            print("? - Show this help message")
        else:
            print(f"Unknown command: {key}")

if __name__ == "__main__":
    main()