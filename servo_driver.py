import time
import serial
from serial.tools import list_ports

#Some of this is reused from
#https://github.com/Gymnast544/DS-Input-Interface-Software
#and
#https://github.com/kdamy/TheWaterboarders/blob/main/teensy_computer_interface/teensy_interface.py

ser = None
def initSerial(comport):
    global ser
    ser = serial.Serial(comport)
    ser.baudrate = 115200
    print("Serial initialized")


def closeSerial():
    global ser
    ser.close()

def chooseDevice():
    serialdevices = []
    comports = list_ports.comports()
    potentialports = []
    for port in comports:
        if port.description != "n/a":
            potentialports.append(port)
    if(len(potentialports)>1):
        for index, serialport in enumerate(potentialports):
            print(str(index+1)+": '"+serialport.description+"'")
            serialdevices.append(serialport.device)
        serialindex = int(input("Choose the serial port to use (Enter the number) "))
        comport = serialdevices[serialindex-1]
    else:
        comport = potentialports[0].device
        print("Port selected as "+str(comport))
    return comport

def sendByte(byteint):
    global ser
    ser.write(bytes(chr(byteint), 'utf-8'))


horizServoPos = 90
def sethorizServo(position):
    #UPDATE THESE LIMITS
    if position>97:
        position = 97
    elif position<86:
        position = 86
    global horizServoPos
    xServoPos = position
    sendByte(73)
    sendByte(position)

panServoPos = 90
def setPanServo(position):
    #UPDATE THESE LIMITS
    if position>120:
        position = 120
    elif position<60:
        position = 60
    global yServoPos
    yServoPos = position
    sendByte(74)
    sendByte(position)

tiltServoPos = 90
def setTiltServo(position):
    #UPDATE THESE LIMITS
    if position>97:
        position = 97
    elif position<86:
        position = 86
    global tiltServoPos
    tiltServoPos = position
    sendByte(75)
    sendByte(position)
extendServoPos = 90
def setExtendServo(position):
    #UPDATE THESE LIMITS
    if position>120:
        position = 120
    elif position<60:
        position = 60
    global extendServoPos
    extendServoPos = position
    sendByte(76)
    sendByte(position)




chooseDevice()