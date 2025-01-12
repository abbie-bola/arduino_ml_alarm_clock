#ifndef RTC_AND_LCD_H
#define RTC_AND_LCD_H
#include <Arduino.h>

// set up functions
void checkRTC(void);
void display_time(void);

// alarm functions
void alarm_setup(void);
void timer_second(int n);
void set_alarm(void);
// bool alarm_status(void); // checks if the alarm has been fired and returns true or false
void reset_alarm(int n);
void stop_alarm(void);

#endif
