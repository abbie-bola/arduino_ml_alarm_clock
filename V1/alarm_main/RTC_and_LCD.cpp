// this file is for the LCD and RTC module
// it is seperate to functions.cpp as it has too many of its own complexities
// RTC and LCD are in one seperate header file together because 
// to display RTC info on the LCD, it will need both libraries, so incorporating them is easier

// here:
// 1. the RTC will be set up to ensure I2c communication works
// 2. the screen will be set up to display the time and date 
// 3. additional functions for alarm functions
//    ^^^ further info about these functions are commented within the functions

#include <LiquidCrystal.h>
#include <RTClib.h>
#include <Wire.h>

#include "RTC_and_LCD.h"

// Creating objects of the LCD and RTC 
LiquidCrystal lcd(2, 3, 4, 5, 6, 7); //Pins: RS, E, D4, D5, D6, D7
RTC_DS3231 rtc; 

// array to convert weekday number to a string 
// dayNames[0] = Sun, 1 = Mon, 2 = Tue...
const char* dayNames[] = {"Sun" , "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

void checkRTC(void) {
  if (!rtc.begin()) { // if I2C communication with RTC is unsuccessful 
    lcd.print("I2C + RTC Error");
    while (1); // Halt if RTC is not found
  }
  if (rtc.lostPower()) { // if RTC has lost power
    lcd.clear(); // clear the screen
    lcd.print("RTC lost power");
    delay(2000);
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__))); // adjust the date and time
  }
}

void alarm_setup(void) {
    lcd.clear();
    // update message to signal it is fine
    lcd.print("RTC + LCD Sucessful"); 
    delay(2000);
    lcd.clear();
    lcd.begin(16,2); // setting up the screen to be 16x2
    
    // stop oscillating signals at SQW Pin
    // otherwise setAlarm1 will fail
    rtc.writeSqwPinMode(DS3231_OFF);

    // set alarm 1, 2 flag to false (so alarm 1, 2 didn't happen so far)
    // if not done, this easily leads to problems, as both register aren't reset on reboot/recompile
    rtc.disableAlarm(1);
    rtc.disableAlarm(2);
    rtc.clearAlarm(1);
    rtc.clearAlarm(2);

    // turn off alarm 2 (in case it isn't off already)
    // again, this isn't done at reboot, so a previously set alarm could easily go overlooked
    rtc.disableAlarm(2);
}

void display_time(void) {
  DateTime now = rtc.now();

  lcd.clear(); // Clear the LCD screen

  // Display the day of the week and time on the first line
  lcd.setCursor(0, 0); // Set cursor to column 0, row 0
  lcd.print(dayNames[now.dayOfTheWeek()]); // Day of the week
  lcd.print(' ');

  // Display the time in 12-hour format
  int hour = now.hour() % 12;
  if (hour == 0) hour = 12; // Adjust for 12-hour format
  if (hour < 10) lcd.print('0'); // Add leading zero for hour if necessary
  lcd.print(hour, DEC);
  lcd.print(':');
  if (now.minute() < 10) lcd.print('0'); // Add leading zero for minute if necessary
  lcd.print(now.minute(), DEC);
  // lcd.print(':');
  // if (now.second() < 10) lcd.print('0'); // Add leading zero for second if necessary
  // lcd.print(now.second(), DEC);
  lcd.print(now.hour() < 12 ? " AM" : " PM"); // Display AM/PM

  // Display the date on the second line
  lcd.setCursor(0, 1); // Set cursor to column 0, row 1
  if (now.day() < 10) lcd.print('0'); // Add leading zero for day if necessary
  lcd.print(now.day(), DEC);
  lcd.print('.');
  if (now.month() < 10) lcd.print('0'); // Add leading zero for month if necessary
  lcd.print(now.month(), DEC);
  lcd.print('.');
  lcd.print(now.year(), DEC);

  delay(100); // Update every second
}

// this sets an alarm n seconds from now
void timer_second(int n) {
    if(!rtc.setAlarm1(rtc.now() + TimeSpan(n), DS3231_A1_Second)) {
        Serial.print("Error, alarm for "); Serial.print(n); Serial.println(" seconds from now was NOT set!");
    }
    else {
        Serial.print("Alarm is set for "); Serial.print(n); Serial.println(" seconds from now!"); 
    }
}

// TODO 
// make an enum or class for modes to use as parameter 
// remake set_alarm function to take in number parameters for the dateTime(...)

// this sets an alarm at a certain time 
void set_alarm(void) {
    if(!rtc.setAlarm1(DateTime(0, 0, 0, 10, 30, 0), DS3231_A1_Hour)) {
        Serial.println("Error, alarm for 10:30AM wasn't set!");
    }
    else {
        Serial.println("Alarm is set for 10:30AM!");
    }
}

void reset_alarm(int n) {
  // there can only be alarm 1 or 2 - other numbers can result in a bug
  if (n == 1 || n == 2) {
    // resetting SQW and alarm 1 flag
    // using setAlarm1, the next alarm could now be configurated
    // parameter int n so that user can chose to reset alarm 1 or 2
    if (rtc.alarmFired(n)) {
        rtc.disableAlarm(n);
        rtc.clearAlarm(n); 
        Serial.println(" - Alarm cleared");
    }
  }
  else {
    Serial.println("Wrong parameter inputted for reset_alarm function in RTC_and_LCD.cpp");
  }
}


// delete later 
void stop_alarm(void) {
  rtc.clearAlarm(1);
  rtc.disableAlarm(1);
  Serial.println("...Alarm cleared");
}

// // bool to check if the alarm has been cleared or not
// bool alarm_status(void) {
//   if (rtc.alarmFired(1)) {
//       Serial.print("Alarm has been fired!!!");
//       return true;
//   }
//   else {
//     return false;
//   }
// }


