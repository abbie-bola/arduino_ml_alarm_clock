// this is the h file which will have the prototypes of the functions
// this header file will also include <Arduino.h> so it will not have to be included in the cpp file either

#ifndef BUZZER_H
#define BUZZER_H
#include <Arduino.h>

void silence(void);
void forever_beep(void);
void play_song(void);
void beep(void);

#endif
