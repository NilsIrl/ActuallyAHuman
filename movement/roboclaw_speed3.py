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

rc.Open()
address = 0x80

version = rc.ReadVersion(address)
if not version[0]:
    print("GETVERSION Failed")
else:
    print(repr(version[1]))


rc.SetM1VelocityPID(address, 100, 0, 0, 2500)
rc.SetM2VelocityPID(address, 100, 0, 0, 2500)

rotation_target = 45
desired_heading = get_latest_imu_data() + rotation_target

start_time = time.time()
PGAIN = 3
# FIXME: use monotonic time
while time.time() - start_time < 10:
    imu_reading = get_latest_imu_data()
    print(f"imu_reading: {imu_reading}")
    angle_difference = desired_heading - imu_reading
    print(f"angle_difference: {angle_difference}")
    if abs(angle_difference) < 2.5:
        break

    turnpower = angle_difference * PGAIN
    turnpower_m2 = -turnpower

    print(f"turnpower: {turnpower}")
    turnpower = max(-64, min(63, turnpower))
    turnpower_m2 = max(-64, min(63, turnpower_m2))

    final_power_m1 = int(turnpower + 64)
    final_power_m2 = int(turnpower_m2 + 64)

    print(f"final_power: {final_power_m1}")
    print(f"final_power: {final_power_m2}")

    rc.ForwardBackwardM1(address, final_power_m1)
    rc.ForwardBackwardM2(address, final_power_m2)

# FIXME: use monotonic time

rc.ForwardBackwardM1(address, 64)
rc.ForwardBackwardM2(address, 64)
