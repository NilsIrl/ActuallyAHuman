#include <Servo.h>

//Arduino mega used
Servo horiz;
Servo pan;
Servo tilt;
Servo extend;

#define HORIZ_PIN 2
#define PAN_PIN 3
#define TILT_PIN 4
#define EXTEND_PIN 5

int servoByte = 0;
void setup() {
  // put your setup code here, to run once:

  horiz.attach(HORIZ_PIN);
  pan.attach(PAN_PIN);
  tilt.attach(TILT_PIN);
  extend.attach(EXTEND_PIN);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < Serial.available(); i++) {
    int incomingbyte = Serial.read();
    if (servoByte == 0) {
      if (incomingbyte == 73) {
        servoByte = 1;
        //Serial.write("Awaiting byte for Horiz");
      } else if (incomingbyte == 74) {
        servoByte = 2;
        //Serial.write("Awaiting byte for Pan");
      }
      else if (incomingbyte == 75) {
        servoByte = 2;
        //Serial.write("Awaiting byte for Tilt");
      }
      else if (incomingbyte == 76) {
        servoByte = 2;
        //Serial.write("Awaiting byte for Extend");
      }
    }else{
      if(servoByte == 1){
        horiz.write(incomingbyte);
        Serial.write("Set horiz to byte");
      }else if(servoByte == 2){
        pan.write(incomingbyte);
        Serial.write("Set pan to byte");
      }else if(servoByte == 3){
        tilt.write(incomingbyte);
        Serial.write("Set tilt to byte");
      }else if(servoByte == 4){
        extend.write(incomingbyte);
        Serial.write("Set extend to byte");
      }
      servoByte = 0;
    }
  }

}