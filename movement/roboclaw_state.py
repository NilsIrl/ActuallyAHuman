import math
import time
from roboclaw import Roboclaw
import sqlite3

# Windows comport name
#rc = Roboclaw("COM3", 115200)
# Linux comport name
#rc = Roboclaw("/dev/ttyACM0", 115200)
#rc = Roboclaw("/dev/serial/by-id/usb-Basicmicro_Inc._USB_Roboclaw_2x15A-if00", 115200)
rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0", 115200)

def displayspeed():
    enc1 = rc.ReadEncM1(address)
    enc2 = rc.ReadEncM2(address)
    speed1 = rc.ReadSpeedM1(address)
    speed2 = rc.ReadSpeedM2(address)

    print("Encoder1:", end=' ')
    if enc1[0] == 1:
        print(enc1[1], format(enc1[2], '02x'))
    else:
        print("failed", end=' ')

    print("Encoder2:", end=' ')
    if enc2[0] == 1:
        print(enc2[1], format(enc2[2], '02x'))
    else:
        print("failed", end=' ')

    print("Speed1:", end=' ')
    if speed1[0]:
        print(speed1[1])
    else:
        print("failed", end=' ')

    print("Speed2:", end=' ')
    if speed2[0]:
        print(speed2[1])
    else:
        print("failed")

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

rc.Open()
address = 0x80

version = rc.ReadVersion(address)
if not version[0]:
    print("GETVERSION Failed")
else:
    print(repr(version[1]))


rc.SetM1PositionPID(address, 200, 0, 4000, 100, 10, -2_000_000_000, 2_000_000_000)
rc.SetM2PositionPID(address, 200, 0, 4000, 100, 10, -2_000_000_000, 2_000_000_000)
rc.SetM1VelocityPID(address, 400, 0, 0, 2500)
rc.SetM2VelocityPID(address, 400, 0, 0, 2500)

NUM_OF_METERS_TO_GO_FORWARD = 1
# In meters
DIAMETER_OF_WHEEL = 0.1524
POSITIONS_PER_REVOLUTION = 1120

# Calculate positions to move based on distance
CIRCUMFERENCE_OF_WHEEL = DIAMETER_OF_WHEEL * math.pi
REVOLUTIONS_NEEDED = NUM_OF_METERS_TO_GO_FORWARD / CIRCUMFERENCE_OF_WHEEL
POSITIONS_TO_INCREASE = int(REVOLUTIONS_NEEDED * POSITIONS_PER_REVOLUTION)


enc1 = rc.ReadEncM1(address)
enc2 = rc.ReadEncM2(address)

assert enc1[0] == 1
assert enc2[0] == 1

print(f"Need to increase position by {POSITIONS_TO_INCREASE} ticks")
m1_target = enc1[1] + POSITIONS_TO_INCREASE
m2_target = enc2[1] + POSITIONS_TO_INCREASE
print(f"m1_target: {m1_target}")
print(f"m2_target: {m2_target}")
# rc.SetM1Position(address, m1_target, 1)
# rc.SetM2Position(address, m2_target, 1)

# rc.BackwardM1(address, 127)
# rc.BackwardM2(address, 127)

# rc.ForwardM1(address, 127)
# rc.ForwardM2(address, 127)

rc.SpeedM1(address, 2500)
rc.SpeedM2(address, 2500)

# rc.SpeedM1M2(address, 127, 127)

time.sleep(1)

rc.ForwardM1(address, 0)
rc.ForwardM2(address, 0)



# rc.SetM1M2Position(address, enc1[1] + POSITIONS_TO_INCREASE, enc2[1] + POSITIONS_TO_INCREASE, 1)


# while True:
#     enc1 = rc.ReadEncM1(address)
#     enc2 = rc.ReadEncM2(address)

#     assert enc1[0] == 1
#     assert enc2[0] == 1

#     print(f"enc1: {enc1[1]}")
#     print(f"enc2: {enc2[1]}")
#     print(f"m1_target: {m1_target}")
#     print(f"m2_target: {m2_target}")
    
#     if enc1[1] >= m1_target or enc2[1] >= m2_target:
#         rc.ForwardM1(address, 0)
#         rc.ForwardM2(address, 0)
#         break
    