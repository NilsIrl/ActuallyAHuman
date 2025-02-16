#include <Servo.h>

//Arduino mega used
Servo pan;
Servo extend;

#define PAN_PIN 2
#define EXTEND_PIN 3

int servoByte = 0;
void setup() {
  // put your setup code here, to run once:

  // pan.write(90 + 30);
  // horiz.attach(HORIZ_PIN);
  pan.attach(PAN_PIN);
  extend.attach(EXTEND_PIN);

  pan.write(90);
  // 100 is "in" position
  // 60 is "out" position
  extend.write(100);

  Serial.begin(9600);  
}

void loop() {
  // put your main code here, to run repeatedly:
  int selector = Serial.read();
  int value = Serial.read();

  switch (selector) {
    case 1:
      pan.write(value);
      break;
    case 2:
      extend.write(value);
      break;
  }
}