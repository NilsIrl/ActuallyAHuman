import serial
import time

class IMUReader:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, timeout=1):
        """
        Initialize connection to Arduino IMU via serial port.
        :param port: Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux/Mac)
        :param baudrate: Baud rate (must match Arduino Serial.begin())
        :param timeout: Read timeout in seconds
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Allow time for connection to establish

    def get_heading(self):
        """
        Reads the latest heading value from the Arduino IMU.
        :return: Floating point heading value
        """
        self.ser.flushInput()  # Clear buffer
        try:
            line = self.ser.readline().decode("utf-8").strip()
            return float(line) if line else None
        except ValueError:
            return None

    def close(self):
        """Close the serial connection."""
        self.ser.close()

if __name__ == "__main__":
    imu = IMUReader(port="/dev/ttyUSB0")  # Adjust port as needed
    while True:
        heading = imu.get_heading()
        if heading is not None:
            print(f"Heading: {heading}°")
        time.sleep(0.1)
