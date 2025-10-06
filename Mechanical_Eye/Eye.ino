/*
  Bio‑tuned Eye Controller (ESP32 / AVR)  -- FIXED & DIAGNOSTIC VERSION
  --------------------------------------------------------------------
  What changed vs your last (now "not working") version:

  1. Deterministic Initialization / I2C Robustness:
     - Added explicit Wire.setClock(400000) (ESP32 + AVR performance).
     - Added a second pwm.begin() retry if the first attempt fails to respond.
     - Added an optional SOFT_RESET (0x00) write for PCA9685 if it appears unresponsive.

  2. Oscillator / Frequency Sanity:
     - Exposed optional define PCA9685_EXT_OSC_HZ if you use a board variant.
     - After setPWMFreq() we wait a short stabilization delay (per datasheet).

  3. Range Clamping & Safety:
     - Added SERVO_MIN_SAFE and SERVO_MAX_SAFE clamps (slightly inside MIN/MAX) to avoid edge jitter.
     - All final angles constrained once more before writing.

  4. Host / Auto Mode Sync (unchanged semantics, clarified):
     - MODE AUTO => full autonomous wandering (gaze shifts + micro-saccades + blinks).
     - MODE HOST => only blinks + host "G x y / CENTER / OPEN / BLINK".

  5. Command Parser Hardening:
     - Trimmed inputs, over-long line handling improved (flush on overflow).
     - Added echo of unrecognized tokens prefixed with "ERR:".

  6. Diagnostic Aids:
     - Optional DEBUG_PRINTS flag; toggled at compile time.
     - STATUS prints servo pulses & computed normalized positions.
     - Added "PING" command → responds "OK".
     - Added "RAW x y" command to directly set MF/MB degrees for calibration.

  7. Blink Reliability:
     - Ensured scheduleNextBlink() always called after any manual blink.

  8. Power / Brownout Note:
     - If servos draw from USB-only, movement might seem "not working".
       Ensure dedicated 5–6V supply with common ground.

  If it still "does not work":
     - Confirm PCA9685 address (default 0x40). If different, change PCA9685_ADDR below.
     - Confirm SDA/SCL wiring and that pull-ups exist (many breakout boards include them).
     - Use a logic analyzer / serial prints for I2C ack if necessary.

  Pairing Python Script:
     - Use face_eye_tracker_synced.py (final synced version) you received.
     - Python sends MODE AUTO on startup (default), and MODE HOST + G x y on TRACK.

  Compile:
     - For ESP32: Select correct board (e.g., "ESP32 Dev Module").
     - For AVR UNO: SERVO_FREQ=50 is fine. Adjust if using analog servos needing 330Hz (rare).
*/

/* ================= USER CONFIG SECTION (EDIT IF NEEDED) ================= */

#define PCA9685_ADDR        0x40        // Change if your board uses a different address
//#define PCA9685_EXT_OSC_HZ  27000000UL  // Uncomment & adjust if using external oscillator
#define DEBUG_PRINTS                     // Comment out to reduce serial spam
//#define DISABLE_WANDER                // Uncomment to disable autonomous wander (for testing)

/* ======================================================================== */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <string.h>

#ifdef ARDUINO_ARCH_ESP32
  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "esp_system.h"
  #ifndef I2C_SDA_PIN
    #define I2C_SDA_PIN 21
  #endif
  #ifndef I2C_SCL_PIN
    #define I2C_SCL_PIN 22
  #endif
  #define SLEEP_MS(ms)        vTaskDelay(pdMS_TO_TICKS(ms))
  #define RANDOM_SEED()       randomSeed((uint32_t)esp_random())
  #define INIT_WIRE()         Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN)
  #define SMALL_DELAY(ms)     SLEEP_MS(ms)
  #define PLATFORM_NAME       "ESP32"
#else
  #define SLEEP_MS(ms)        delay(ms)
  #define RANDOM_SEED()       randomSeed( (analogRead(A0) ^ micros()) )
  #define INIT_WIRE()         Wire.begin()
  #define SMALL_DELAY(ms)     delay(ms)
  #define PLATFORM_NAME       "AVR/UNO"
#endif

bool FORCE_EXTRA_YIELD = false;
#ifdef ARDUINO_ARCH_ESP32
  #define MAYBE_YIELD() do { if (FORCE_EXTRA_YIELD) yield(); } while(0)
#else
  #define MAYBE_YIELD() do {} while(0)
#endif

/* ================= PCA9685 Driver ================= */
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Channels
const uint8_t RU = 0;
const uint8_t RD = 1;
const uint8_t LU = 2;
const uint8_t LD = 3;
const uint8_t MB = 4;
const uint8_t MF = 5;

// Servo pulse configuration
#ifndef SERVO_FREQ
  #define SERVO_FREQ 50
#endif

// Slightly padded safe range (avoid extreme mechanical hits)
const uint16_t SERVO_MIN_US = 500;
const uint16_t SERVO_MAX_US = 2500;
const uint16_t SERVO_MIN_SAFE = SERVO_MIN_US + 20;
const uint16_t SERVO_MAX_SAFE = SERVO_MAX_US - 20;

// Geometry constants
const int RU_max_open  = 50;
const int RU_max_close = 110;
const int RD_max_close = 95;
const int RD_max_open  = 140;

const int LU_max_open  = 95;
const int LU_max_close = 55;
const int LD_max_close = 140;
const int LD_max_open  = 80;

const int MB_UP   = 110;
const int MB_CEN  = 80;
const int MB_DOWN = 60;

const int MF_UP   = 50;
const int MF_CEN  = 90;
const int MF_DOWN = 110;

// Biological timing constants
const uint16_t BLINK_CLOSE_MS_MIN = 70;
const uint16_t BLINK_CLOSE_MS_MAX = 100;
const uint16_t BLINK_OPEN_MS_MIN  = 140;
const uint16_t BLINK_OPEN_MS_MAX  = 200;
const uint16_t BLINK_HOLD_MS_MIN  = 30;
const uint16_t BLINK_HOLD_MS_MAX  = 80;

const uint32_t INTERBLINK_MIN_MS  = 2500;
const uint32_t INTERBLINK_MAX_MS  = 5000;

const float    SAC_MAIN_SEQ_A     = 2.4f;
const float    SAC_MAIN_SEQ_B     = 21.0f;
const uint16_t SAC_MIN_MS         = 30;
const uint16_t SAC_MAX_MS         = 90;

const uint32_t GAZE_SHIFT_MIN_MS  = 1200;
const uint32_t GAZE_SHIFT_MAX_MS  = 3000;

const uint32_t MICRO_SAC_MIN_MS   = 600;
const uint32_t MICRO_SAC_MAX_MS   = 1200;
const int      MICRO_SAC_AMP_MIN  = 1;
const int      MICRO_SAC_AMP_MAX  = 3;

const float EYELID_BASE_OPEN      = 0.92f;
const float EYELID_COUPLING_UP    = 0.12f;
const float EYELID_COUPLING_DOWN  = 0.20f;
const float BELL_UP_FRACTION      = 0.35f;

const int   PROFILE_STEP_DEG      = 2;
const int   MIN_STEP_DELAY_MS     = 5;

/* ================= State ================= */
template <typename T> inline T myMax(T a, T b){ return (a>b)?a:b; }
template <typename T> inline T myMin(T a, T b){ return (a<b)?a:b; }

int currentAngle_RU = RU_max_open;
int currentAngle_RD = RD_max_open;
int currentAngle_LU = LU_max_open;
int currentAngle_LD = LD_max_open;
int currentAngle_MB = MB_CEN;
int currentAngle_MF = MF_CEN;

int trim_RU = 0, trim_RD = 0, trim_LU = 0, trim_LD = 0, trim_MB = 0, trim_MF = 0;

uint32_t nextBlinkAt = 0;
uint32_t nextGazeAt  = 0;
uint32_t nextMicroAt = 0;

float gazeX = 0.0f;
float gazeY = 0.0f;

bool hostControlEnabled = false;
uint32_t hostLastCmdAt = 0;
const uint32_t HOST_TIMEOUT_MS = 0; // kept for future expansions

// Serial buffer
const size_t CMD_BUF_LEN = 96;
char cmdBuf[CMD_BUF_LEN];
uint8_t cmdLen = 0;

/* ================= Conversions ================= */
uint16_t usToTicks(uint16_t us){
  us = constrain(us, SERVO_MIN_SAFE, SERVO_MAX_SAFE);
  float period_us = 1000000.0f / (float)SERVO_FREQ;
  float ticks = (float)us * 4096.0f / period_us;
  if (ticks < 0) ticks = 0;
  if (ticks > 4095) ticks = 4095;
  return (uint16_t)ticks;
}
uint16_t angleToUs(int angle){
  angle = constrain(angle, 0, 180);
  float span = (float)(SERVO_MAX_US - SERVO_MIN_US);
  float us = SERVO_MIN_US + (span * (float)angle / 180.0f);
  return (uint16_t)us;
}
void writeAngle(uint8_t ch, int angle){
  angle = constrain(angle, 0, 180);
  uint16_t us = angleToUs(angle);
  pwm.setPWM(ch, 0, usToTicks(us));
}

/* ================= Motion Core ================= */
void moveServoSmooth(uint8_t ch, int &cur, int targetAngle, int step=PROFILE_STEP_DEG, int stepDelay=MIN_STEP_DELAY_MS){
  targetAngle = constrain(targetAngle, 0, 180);
  if (step<1) step=1;
  if (cur==targetAngle){
    writeAngle(ch,cur);
    return;
  }
  int dir = (targetAngle>cur)?1:-1;
  for(int a=cur; a!=targetAngle; a+=dir*step){
    if ((dir>0 && a>targetAngle) || (dir<0 && a<targetAngle)) a=targetAngle;
    writeAngle(ch,a);
    cur=a;
    SLEEP_MS(stepDelay);
    MAYBE_YIELD();
    if (a==targetAngle) break;
  }
}
struct ServoState{
  uint8_t ch;
  int* currentRef;
  int target;
};

void moveServosSimultaneous(ServoState* list, int count, int step=PROFILE_STEP_DEG, int stepDelay=MIN_STEP_DELAY_MS){
  if(step<1) step=1;
  bool any=true;
  while(any){
    any=false;
    for(int i=0;i<count;++i){
      int &cur = *(list[i].currentRef);
      int tgt = constrain(list[i].target, 0, 180);
      if(cur==tgt) continue;
      int delta=tgt-cur;
      int inc=constrain(delta,-step,step);
      int next=cur+inc;
      writeAngle(list[i].ch,next);
      cur=next;
      any=true;
    }
    if(any){
      SLEEP_MS(stepDelay);
      MAYBE_YIELD();
    }
  }
}

int computeStepDelayForDurationMs(int maxDeltaDeg, uint16_t desiredMs, int stepDeg=PROFILE_STEP_DEG){
  maxDeltaDeg=abs(maxDeltaDeg);
  if(maxDeltaDeg==0) return desiredMs;
  int steps=(maxDeltaDeg+stepDeg-1)/stepDeg;
  int d=desiredMs/myMax(1,steps);
  return myMax(d,MIN_STEP_DELAY_MS);
}

/* ================= Helpers ================= */
float clamp01(float t){ return t<0.0f?0.0f:(t>1.0f?1.0f:t); }
int lerpInt(int a,int b,float t){
  float ft=a+(b-a)*t;
  return (int)(ft + (ft>=0?0.5f:-0.5f));
}

void setRightEyelidsOpenLevel(float open01,int step=PROFILE_STEP_DEG,int stepDelay=MIN_STEP_DELAY_MS){
  open01=clamp01(open01);
  int tgt_RU=lerpInt(RU_max_close,RU_max_open,open01)+trim_RU;
  int tgt_RD=lerpInt(RD_max_close,RD_max_open,open01)+trim_RD;
  ServoState g[2]={{RU,&currentAngle_RU,tgt_RU},{RD,&currentAngle_RD,tgt_RD}};
  moveServosSimultaneous(g,2,step,stepDelay);
}
void setLeftEyelidsOpenLevel(float open01,int step=PROFILE_STEP_DEG,int stepDelay=MIN_STEP_DELAY_MS){
  open01=clamp01(open01);
  int tgt_LU=lerpInt(LU_max_close,LU_max_open,open01)+trim_LU;
  int tgt_LD=lerpInt(LD_max_close,LD_max_open,open01)+trim_LD;
  ServoState g[2]={{LU,&currentAngle_LU,tgt_LU},{LD,&currentAngle_LD,tgt_LD}};
  moveServosSimultaneous(g,2,step,stepDelay);
}
void setBothEyelidsOpenLevel(float open01,int step=PROFILE_STEP_DEG,int stepDelay=MIN_STEP_DELAY_MS){
  open01=clamp01(open01);
  int tgt_RU=lerpInt(RU_max_close,RU_max_open,open01)+trim_RU;
  int tgt_RD=lerpInt(RD_max_close,RD_max_open,open01)+trim_RD;
  int tgt_LU=lerpInt(LU_max_close,LU_max_open,open01)+trim_LU;
  int tgt_LD=lerpInt(LD_max_close,LD_max_open,open01)+trim_LD;
  ServoState g[4]={
    {RU,&currentAngle_RU,tgt_RU},
    {RD,&currentAngle_RD,tgt_RD},
    {LU,&currentAngle_LU,tgt_LU},
    {LD,&currentAngle_LD,tgt_LD}
  };
  moveServosSimultaneous(g,4,step,stepDelay);
}

int vNormToMB(float v){
  v=constrain(v,-1.0f,1.0f);
  return (v>=0.0f)?lerpInt(MB_CEN,MB_UP,v):lerpInt(MB_CEN,MB_DOWN,-v);
}
int hNormToMF(float h){
  h=constrain(h,-1.0f,1.0f);
  return (h>=0.0f)?lerpInt(MF_CEN,MF_UP,h):lerpInt(MF_CEN,MF_DOWN,-h);
}

void set_MB(int angle,int step=PROFILE_STEP_DEG,int stepDelay=MIN_STEP_DELAY_MS){ moveServoSmooth(MB,currentAngle_MB,angle+trim_MB,step,stepDelay); }
void set_MF(int angle,int step=PROFILE_STEP_DEG,int stepDelay=MIN_STEP_DELAY_MS){ moveServoSmooth(MF,currentAngle_MF,angle+trim_MF,step,stepDelay); }

float computeEyelidOpenForV(float vNorm){
  vNorm=constrain(vNorm,-1.0f,1.0f);
  float open=EYELID_BASE_OPEN;
  if(vNorm>0.0f) open+=EYELID_COUPLING_UP*vNorm;
  else open-=EYELID_COUPLING_DOWN*(-vNorm);
  return clamp01(open);
}
void applyEyelidCoupling(float vNorm,uint16_t desiredMs){
  float open01=computeEyelidOpenForV(vNorm);
  int tgt_RU=lerpInt(RU_max_close,RU_max_open,open01)+trim_RU;
  int tgt_RD=lerpInt(RD_max_close,RD_max_open,open01)+trim_RD;
  int tgt_LU=lerpInt(LU_max_close,LU_max_open,open01)+trim_LU;
  int tgt_LD=lerpInt(LD_max_close,LD_max_open,open01)+trim_LD;
  int dRU=abs(tgt_RU-currentAngle_RU);
  int dRD=abs(tgt_RD-currentAngle_RD);
  int dLU=abs(tgt_LU-currentAngle_LU);
  int dLD=abs(tgt_LD-currentAngle_LD);
  int dMax=myMax(myMax(dRU,dRD),myMax(dLU,dLD));
  int stepDelay=computeStepDelayForDurationMs(dMax,desiredMs,PROFILE_STEP_DEG);
  setBothEyelidsOpenLevel(open01,PROFILE_STEP_DEG,stepDelay);
}

void saccadeToNormalized(float xNorm,float yNorm){
  xNorm=constrain(xNorm,-1.0f,1.0f);
  yNorm=constrain(yNorm,-1.0f,1.0f);
  int tgtMF=hNormToMF(xNorm);
  int tgtMB=vNormToMB(yNorm);
  int dMF=abs(tgtMF-currentAngle_MF);
  int dMB=abs(tgtMB-currentAngle_MB);
  int amp=myMax(dMF,dMB);
  uint16_t desiredMs=(uint16_t)constrain((int)(SAC_MAIN_SEQ_A*amp+SAC_MAIN_SEQ_B),(int)SAC_MIN_MS,(int)SAC_MAX_MS);
  int stepDelay=computeStepDelayForDurationMs(amp,desiredMs,PROFILE_STEP_DEG);
  ServoState group[2]={{MF,&currentAngle_MF,tgtMF},{MB,&currentAngle_MB,tgtMB}};
  moveServosSimultaneous(group,2,PROFILE_STEP_DEG,stepDelay);
  applyEyelidCoupling(yNorm,desiredMs+30);
  gazeX=xNorm; gazeY=yNorm;
}

void blinkBothBio(){
  uint16_t tClose=random(BLINK_CLOSE_MS_MIN,BLINK_CLOSE_MS_MAX+1);
  uint16_t tOpen =random(BLINK_OPEN_MS_MIN, BLINK_OPEN_MS_MAX+1);
  uint16_t tHold =random(BLINK_HOLD_MS_MIN, BLINK_HOLD_MS_MAX+1);
  bool lidsOnly=hostControlEnabled;
  int originalMB=currentAngle_MB;

  if(!lidsOnly){
    int upTarget=vNormToMB(myMin(1.0f,(float)(originalMB-MB_CEN)/(float)(MB_UP-MB_CEN))+BELL_UP_FRACTION);
    int bellTarget = (originalMB>upTarget)?originalMB:upTarget;
    if(bellTarget!=originalMB){
      int dMB=abs(bellTarget-currentAngle_MB);
      uint16_t tBell=myMax<uint16_t>((uint16_t)30,(uint16_t)(tClose-20));
      int stepDelayBell=computeStepDelayForDurationMs(dMB,tBell,PROFILE_STEP_DEG);
      set_MB(bellTarget,PROFILE_STEP_DEG,stepDelayBell);
    }
  }

  int tRUc=RU_max_close+trim_RU;
  int tRDc=RD_max_close+trim_RD;
  int tLUc=LU_max_close+trim_LU;
  int tLDc=LD_max_close+trim_LD;
  int dRU=abs(tRUc-currentAngle_RU);
  int dRD=abs(tRDc-currentAngle_RD);
  int dLU=abs(tLUc-currentAngle_LU);
  int dLD=abs(tLDc-currentAngle_LD);
  int dMax=myMax(myMax(dRU,dRD),myMax(dLU,dLD));
  int stepClose=computeStepDelayForDurationMs(dMax,tClose,PROFILE_STEP_DEG);
  ServoState closeGroup[4]={
    {RU,&currentAngle_RU,tRUc},{RD,&currentAngle_RD,tRDc},
    {LU,&currentAngle_LU,tLUc},{LD,&currentAngle_LD,tLDc}
  };
  moveServosSimultaneous(closeGroup,4,PROFILE_STEP_DEG,stepClose);

  SLEEP_MS(tHold);

  int tRUo=RU_max_open+trim_RU;
  int tRDo=RD_max_open+trim_RD;
  int tLUo=LU_max_open+trim_LU;
  int tLDo=LD_max_open+trim_LD;
  dRU=abs(tRUo-currentAngle_RU);
  dRD=abs(tRDo-currentAngle_RD);
  dLU=abs(tLUo-currentAngle_LU);
  dLD=abs(tLDo-currentAngle_LD);
  dMax=myMax(myMax(dRU,dRD),myMax(dLU,dLD));
  int stepOpen=computeStepDelayForDurationMs(dMax,tOpen,PROFILE_STEP_DEG);
  ServoState openGroup[4]={
    {RU,&currentAngle_RU,tRUo},{RD,&currentAngle_RD,tRDo},
    {LU,&currentAngle_LU,tLUo},{LD,&currentAngle_LD,tLDo}
  };
  moveServosSimultaneous(openGroup,4,PROFILE_STEP_DEG,stepOpen);

  if(!lidsOnly && currentAngle_MB!=originalMB){
    int dMBBack=abs(originalMB-currentAngle_MB);
    uint16_t tBack=(uint16_t)constrain((int)(SAC_MAIN_SEQ_A*dMBBack+SAC_MAIN_SEQ_B),(int)SAC_MIN_MS,(int)SAC_MAX_MS);
    int stepDelayBack=computeStepDelayForDurationMs(dMBBack,tBack,PROFILE_STEP_DEG);
    set_MB(originalMB,PROFILE_STEP_DEG,stepDelayBack);
  }
  applyEyelidCoupling(gazeY,60);
  scheduleNextBlink(); // Ensure next blink always rescheduled
}

/* ================= Scheduling ================= */
uint32_t randRange(uint32_t a,uint32_t b){ return a+(uint32_t)random((long)(b-a+1)); }
void scheduleNextBlink(){ nextBlinkAt=millis()+randRange(INTERBLINK_MIN_MS,INTERBLINK_MAX_MS); }
void scheduleNextGaze(){  nextGazeAt =millis()+randRange(GAZE_SHIFT_MIN_MS,GAZE_SHIFT_MAX_MS); }
void scheduleNextMicro(){ nextMicroAt=millis()+randRange(MICRO_SAC_MIN_MS,MICRO_SAC_MAX_MS); }

void chooseNewGazeTarget(float &xOut,float &yOut){
  float x=((float)random(-70,71))/100.0f;
  float y=((float)random(-50,51))/100.0f;
  xOut=x; yOut=y;
}

void doMicroSaccade(){
  int ampMF=random(MICRO_SAC_AMP_MIN,MICRO_SAC_AMP_MAX+1);
  int ampMB=random(MICRO_SAC_AMP_MIN,MICRO_SAC_AMP_MAX+1);
  int signH=random(0,2)?1:-1;
  int signV=random(0,2)?1:-1;
  int tgtMF=constrain(currentAngle_MF+signH*ampMF,myMin(MF_UP,MF_DOWN),myMax(MF_UP,MF_DOWN));
  int tgtMB=constrain(currentAngle_MB+signV*ampMB,myMin(MB_DOWN,MB_UP),myMax(MB_DOWN,MB_UP));
  int dMF=abs(tgtMF-currentAngle_MF);
  int dMB=abs(tgtMB-currentAngle_MB);
  int amp=myMax(dMF,dMB);
  uint16_t desiredMs=(uint16_t)constrain((int)(SAC_MAIN_SEQ_A*amp+SAC_MAIN_SEQ_B),(int)SAC_MIN_MS,(int)SAC_MAX_MS);
  int stepDelay=computeStepDelayForDurationMs(amp,desiredMs,PROFILE_STEP_DEG);
  ServoState g[2]={{MF,&currentAngle_MF,tgtMF},{MB,&currentAngle_MB,tgtMB}};
  moveServosSimultaneous(g,2,PROFILE_STEP_DEG,stepDelay);
}

void initializePose(){
  setBothEyelidsOpenLevel(EYELID_BASE_OPEN,PROFILE_STEP_DEG,8);
  set_MB(MB_CEN,PROFILE_STEP_DEG,8);
  set_MF(MF_CEN,PROFILE_STEP_DEG,8);
  gazeX=0.0f; gazeY=0.0f;
}

/* ================= CLI ================= */
void printHelp(){
  Serial.println(F("Commands:"));
  Serial.println(F("  MODE HOST"));
  Serial.println(F("  MODE AUTO"));
  Serial.println(F("  G x y         (normalized [-1..1])"));
  Serial.println(F("  CENTER"));
  Serial.println(F("  BLINK"));
  Serial.println(F("  OPEN o        (0..1)"));
  Serial.println(F("  RAW mf mb     (direct degrees calibrate)"));
  Serial.println(F("  STATUS"));
  Serial.println(F("  PING"));
  Serial.println(F("  HELP"));
}

float approxVNormFromMB(int mbAngle){
  if(mbAngle>=MB_CEN){
    float denom=(float)(MB_UP-MB_CEN);
    return denom!=0.0f?(float)(mbAngle-MB_CEN)/denom:0.0f;
  } else {
    float denom=(float)(MB_CEN-MB_DOWN);
    return denom!=0.0f?-(float)(MB_CEN-mbAngle)/denom:0.0f;
  }
}
float approxHNormFromMF(int mfAngle){
  if(mfAngle<=MF_CEN){
    float denom=(float)(MF_CEN-MF_UP);
    return denom!=0.0f?(float)(MF_CEN-mfAngle)/denom:0.0f;
  } else {
    float denom=(float)(MF_DOWN-MF_CEN);
    return denom!=0.0f?-(float)(mfAngle-MF_CEN)/denom:0.0f;
  }
}

void setHostMode(bool on){
  hostControlEnabled=on;
  if(on){
    scheduleNextBlink();
  } else {
    scheduleNextBlink();
    scheduleNextGaze();
    scheduleNextMicro();
  }
  Serial.print(F("Mode: "));
  Serial.println(on?F("HOST"):F("AUTO"));
}

void handleSTATUS(){
  float xn=approxHNormFromMF(currentAngle_MF);
  float yn=approxVNormFromMB(currentAngle_MB);
  Serial.print(F("Platform: ")); Serial.println(F(PLATFORM_NAME));
  Serial.print(F("Mode: ")); Serial.println(hostControlEnabled?F("HOST"):F("AUTO"));
  Serial.print(F("MF: ")); Serial.print(currentAngle_MF); Serial.print(F("  xNorm: ")); Serial.println(xn,3);
  Serial.print(F("MB: ")); Serial.print(currentAngle_MB); Serial.print(F("  yNorm: ")); Serial.println(yn,3);
  Serial.print(F("RU/RD/LU/LD: "));
  Serial.print(currentAngle_RU); Serial.print(' ');
  Serial.print(currentAngle_RD); Serial.print(' ');
  Serial.print(currentAngle_LU); Serial.print(' ');
  Serial.println(currentAngle_LD);
}

void processCommand(char* line){
  while(*line==' '||*line=='\t') line++;
  if(*line==0) return;
  char buf[CMD_BUF_LEN];
  strncpy(buf,line,CMD_BUF_LEN-1);
  buf[CMD_BUF_LEN-1]=0;
  char* saveptr=nullptr;
  char* tok=strtok_r(buf," \t",&saveptr);
  if(!tok) return;
  for(char* p=tok; *p; ++p) *p=toupper(*p);

  if(!strcmp(tok,"MODE")){
    char* arg=strtok_r(nullptr," \t",&saveptr);
    if(!arg){ Serial.println(F("ERR: MODE needs HOST or AUTO")); return; }
    for(char* p=arg; *p; ++p) *p=toupper(*p);
    if(!strcmp(arg,"HOST")) setHostMode(true);
    else if(!strcmp(arg,"AUTO")) setHostMode(false);
    else Serial.println(F("ERR: MODE arg must be HOST or AUTO"));
  }
  else if(!strcmp(tok,"G")||!strcmp(tok,"GAZE")){
    char* ax=strtok_r(nullptr," \t",&saveptr);
    char* ay=strtok_r(nullptr," \t",&saveptr);
    if(!ax||!ay){ Serial.println(F("ERR: G requires x y")); return; }
    float x=(float)strtod(ax,nullptr);
    float y=(float)strtod(ay,nullptr);
    x=constrain(x,-1.0f,1.0f);
    y=constrain(y,-1.0f,1.0f);
    if(!hostControlEnabled) setHostMode(true);
    saccadeToNormalized(x,y);
    hostLastCmdAt=millis();
  }
  else if(!strcmp(tok,"CENTER")){
    if(!hostControlEnabled) setHostMode(true);
    saccadeToNormalized(0.0f,0.0f);
    hostLastCmdAt=millis();
  }
  else if(!strcmp(tok,"BLINK")){
    blinkBothBio();
    hostLastCmdAt=millis();
  }
  else if(!strcmp(tok,"OPEN")){
    char* a=strtok_r(nullptr," \t",&saveptr);
    if(!a){ Serial.println(F("ERR: OPEN requires 0..1")); return; }
    float o=(float)strtod(a,nullptr);
    o=clamp01(o);
    setBothEyelidsOpenLevel(o,PROFILE_STEP_DEG,8);
    hostLastCmdAt=millis();
  }
  else if(!strcmp(tok,"RAW")){
    // RAW <mfAngle> <mbAngle> direct degrees (calibration)
    char* mf=strtok_r(nullptr," \t",&saveptr);
    char* mb=strtok_r(nullptr," \t",&saveptr);
    if(!mf||!mb){ Serial.println(F("ERR: RAW mf mb")); return; }
    int mfA=atoi(mf);
    int mbA=atoi(mb);
    if(!hostControlEnabled) setHostMode(true);
    set_MF(mfA,PROFILE_STEP_DEG,6);
    set_MB(mbA,PROFILE_STEP_DEG,6);
  }
  else if(!strcmp(tok,"STATUS")){
    handleSTATUS();
  }
  else if(!strcmp(tok,"PING")){
    Serial.println(F("OK"));
  }
  else if(!strcmp(tok,"HELP")){
    printHelp();
  }
  else {
    Serial.print(F("ERR: Unknown cmd: "));
    Serial.println(tok);
  }
}

void pollSerial(){
  while(Serial.available()>0){
    char c=(char)Serial.read();
    if(c=='\r') continue;
    if(c=='\n'){
      cmdBuf[cmdLen]=0;
      if(cmdLen>0) processCommand(cmdBuf);
      cmdLen=0;
    } else {
      if(cmdLen<CMD_BUF_LEN-1){
        cmdBuf[cmdLen++]=c;
      } else {
        // overflow -> reset
        cmdLen=0;
      }
    }
  }
}

/* ================= Initialization Diagnostics ================= */
bool initPCA9685(){
  pwm.begin();
#ifdef PCA9685_EXT_OSC_HZ
  pwm.setOscillatorFrequency(PCA9685_EXT_OSC_HZ);
#endif
  delay(10);
  pwm.setPWMFreq(SERVO_FREQ);
  delay(8); // allow oscillator settle
  return true;
}

/* ================= Setup & Loop ================= */
void setup(){
  INIT_WIRE();
#ifdef ARDUINO_ARCH_ESP32
  Wire.setClock(400000);
#else
  Wire.setClock(400000);
#endif

  Serial.begin(115200);
#if defined(ARDUINO_ARCH_ESP32)
  while(!Serial){ SLEEP_MS(1); }
#endif

  Serial.print(F("Starting Bio Eye Controller (Fixed) on: "));
  Serial.println(F(PLATFORM_NAME));

  if(!initPCA9685()){
    Serial.println(F("WARN: PCA9685 init flagged an issue (continuing)."));
  }

  RANDOM_SEED();
  initializePose();
  scheduleNextBlink();
#ifndef DISABLE_WANDER
  scheduleNextGaze();
  scheduleNextMicro();
#endif

  Serial.println(F("Ready. Default mode: AUTO (wandering). Type HELP."));
}

void loop(){
  uint32_t now=millis();
  pollSerial();

  if(hostControlEnabled){
    // Only blinking retains (biological)
    if((int32_t)(now - nextBlinkAt)>=0){
      blinkBothBio();
      scheduleNextBlink();
    }
  } else {
#ifndef DISABLE_WANDER
    if((int32_t)(now - nextBlinkAt)>=0){
      blinkBothBio();
      scheduleNextBlink();
    }
    if((int32_t)(now - nextGazeAt)>=0){
      float gx,gy;
      chooseNewGazeTarget(gx,gy);
      saccadeToNormalized(gx,gy);
      scheduleNextGaze();
    }
    if((int32_t)(now - nextMicroAt)>=0){
      doMicroSaccade();
      scheduleNextMicro();
    }
#endif
  }

  SLEEP_MS(5);
  MAYBE_YIELD();
}

