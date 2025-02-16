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

rc.Open()
address = 0x80

version = rc.ReadVersion(address)
if not version[0]:
    print("GETVERSION Failed")
else:
    print(repr(version[1]))


rc.ForwardM1(address, 0)
rc.ForwardM2(address, 0)

# while True:
#     print(rc.ReadEncM1(address))
#     print(rc.ReadEncM2(address))
