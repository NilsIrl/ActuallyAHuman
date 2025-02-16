import time
from roboclaw import Roboclaw

# the height from top to bottom is 11979
# as the arm goes down, the position goes down

# rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.4:1.0", 115200)
rc = Roboclaw("/dev/serial/by-path/platform-3610000.usb-usb-0:2.2:1.0", 115200)
rc.Open()

address = 0x80

version = rc.ReadVersion(address)
print(version)

# Make sure the encoder is at 0
# This should be when the head is at the bottom
rc.ResetEncoders(address)
rc.SetM2VelocityPID(address, 0, 0, 0, 3000)
rc.SetM2PositionPID(address, 200, 0, 4000, 100, 10, 0, 10500)

rc.ForwardM2(address, 0)
while True:
    enc2 = rc.ReadEncM2(address)
    print(enc2)
    desired_pos = int(input())
    rc.SetM2Position(address, desired_pos, 1)
