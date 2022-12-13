#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#define PIN        6
#define NUMPIXELS 32

//Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB);
#define DELAYVAL 25

int brightness = 0;
int incomingByte = 0; // for incoming serial data

void setup() {
#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
#endif

  Serial.begin(9600);

  pixels.begin();
  pixels.setBrightness(1); //adjust brightness here
  
  for(int i=0; i<NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(255, 255, 255));
  }
  
  pixels.show();
}

void loop() {
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    brightness =  incomingByte;
    pixels.setBrightness(brightness);
    pixels.show();
  }
}
