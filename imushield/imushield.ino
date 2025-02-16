#include "Arduino_NineAxesMotion.h"
#include <Wire.h>

NineAxesMotion mySensor;
unsigned long lastStreamTime = 0;
const int streamPeriod = 20;  

float prevHeading = 0.0;
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
  if ((millis() - lastStreamTime) >= streamPeriod)
  {
    lastStreamTime = millis();
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

    Serial.println(continuousHeading);  // Output only the heading
  }
}
