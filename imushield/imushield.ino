#include "Arduino_NineAxesMotion.h"
#include <Wire.h>

NineAxesMotion mySensor;

float prevHeading = 0.0;
// heading that doesn't wrap around 360.0
float continuousHeading = 0.0;

void setup() 
{
  Serial.begin(9600);
  Wire.begin();
  
  mySensor.initSensor();
  mySensor.setOperationMode(OPERATION_MODE_NDOF);
  mySensor.setUpdateMode(MANUAL);
}

void loop() 
{
  mySensor.updateEuler();

  float heading = mySensor.readEulerHeading();

  float delta = heading - prevHeading;
  if (delta > 180.0) 
  {
    continuousHeading -= 360.0;
  } 
  else if (delta < -180.0) 
  {
    continuousHeading += 360.0;
  }

  continuousHeading += delta;
  prevHeading = heading;

  Serial.println(continuousHeading);
  delay(30);
}
