import time
from roboclaw import Roboclaw

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


rc.SetM1VelocityPID(address, 100, 0, 0, 2500)
rc.SetM2VelocityPID(address, 100, 0, 0, 2500)

# print(rc.ReadM1VelocityPID(address))
# print(rc.ReadM2VelocityPID(address))

# rc.SetM1VelocityPID(address, 0, 0, 0, 0)
# rc.SetM2PositionPID(address, 0, 0, 0, 0)

displayspeed()
# rc.ForwardM2(address, 127)
# rc.SpeedM2(address, 2500)
# rc.SpeedM1(address, 2500)
rc.SpeedM1M2(address, 2500, 2500)
# rc.ForwardMixed(address, 127)
# rc.ForwardMixed(address, 64)
# rc.TurnLeftMixed(address, 64)
# time.sleep(0.25)
# rc.TurnLeftMixed(address, 127)
# rc.ForwardMixed(address, 127)
time.sleep(2)
# rc.ForwardMixed(address, 0)
# rc.ForwardMixed(address, 0)
# rc.TurnLeftMixed(address, 0)

rc.SpeedM1M2(address, 0, 0)
# rc.SpeedM2(address, 0)
# rc.SpeedM1(address, 0)
# rc.ForwardM2(address, 0)

#while True:
#    rc.SpeedM1(address, 12000)
#    rc.SpeedM2(address, -12000)
#    for _ in range(200):
#        displayspeed()
#        time.sleep(0.01)
#
#    rc.SpeedM1(address, -12000)
#    rc.SpeedM2(address, 12000)
#    for _ in range(200):
#        displayspeed()
#        time.sleep(0.01)
#
displayspeed()
