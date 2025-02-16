import math
import time
from roboclaw import Roboclaw
from microservices_utils import rotate_robot, move_robot_forward_time
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

# move_robot_forward_time(rc, 1, address)
rotate_robot(rc, -90, address)
# move_robot_forward_time(rc, 10, address)