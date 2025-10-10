/*
  Nilheim Mechatronics Servo Heart – REMOTE-SYNC
  - Adds serial control:
      HR <bpm>         : set target BPM (smoothly)
      HRMODE <mode>    : none|brady|tachy|af  (adjusts jitter/character)
      MODE POT         : read potentiometer (default fallback)
      MODE REMOTE      : force remote HR even if idle
      STATUS | PING | HELP
  - Auto mode: if remote is active but no HR arrives for 2000ms, fall back to POT.
*/

#include <Servo.h>
#include <math.h>
#include <string.h>

// ============ Servo pins ============
const int SERVO_ATRIA_L_PIN = 3;
const int SERVO_ATRIA_R_PIN = 5;
const int SERVO_VENT_PIN    = 6;
const int POT_PIN           = A0;

// ============ Servo objects ============
Servo servoAtriaL;
Servo servoAtriaR;
Servo servoVent;

// ============ Configuration ============
constexpr int BPM_MIN = 45;
constexpr int BPM_MAX = 160;

constexpr float POT_EMA_ALPHA = 0.08f;
constexpr uint32_t POT_UPDATE_MS = 60;

constexpr uint32_t SERVO_UPDATE_MS = 15;
constexpr uint32_t POWER_ON_DELAY_MS = 1500;
constexpr uint32_t SOFTSTART_RAMP_MS = 1200;

constexpr uint32_t ATRIA_VENT_GAP_MS = 30;
constexpr float DIASTOLIC_MIN_FRAC = 0.18f;
constexpr uint32_t DIASTOLIC_MIN_MS_LO = 120;
constexpr uint32_t DIASTOLIC_MIN_MS_HI = 400;

// Ventricle geometry (deg)
constexpr float VENT_BASE_DEG = 90.0f;
constexpr float VENT_CONTRACT_DEG = 130.0f;
constexpr float VENT_SAFE_MIN_DEG = 70.0f;
constexpr float VENT_SAFE_MAX_DEG = 130.0f;

// easing functions
inline float easeInCubic(float x)  { if (x<=0) return 0; if (x>=1) return 1; return x*x*x; }
inline float easeOutCubic(float x) { if (x<=0) return 0; if (x>=1) return 1; float y=1.0f-x; return 1.0f - y*y*y; }
inline float easeOutQuint(float x) { if (x<=0) return 0; if (x>=1) return 1; return 1.0f - powf(1.0f - x, 5.0f); }
inline float easeSmoothStep(float x){ if (x<=0) return 0; if (x>=1) return 1; return x*x*(3.0f - 2.0f*x); }
inline float clampf(float v,float lo,float hi){return (v<lo)?lo:((v>hi)?hi:v);}
inline uint32_t clampu32(uint32_t v,uint32_t lo,uint32_t hi){return (v<lo)?lo:((v>hi)?hi:v);}
inline float linProgress(uint32_t now,uint32_t t0,uint32_t dur){
  if(now<=t0)return 0;uint32_t dt=now-t0;if(dt>=dur)return 1;return (float)dt/(float)dur;
}

// servo calibration
struct ServoCal {
  float relaxedAngleDeg;
  float contractedAngleDeg;
};
ServoCal cal[3] = {
  {25.0f, 55.0f}, // left atria
  {25.0f, 55.0f}, // right atria
  {VENT_BASE_DEG, VENT_CONTRACT_DEG} // ventricle
};

// cycle timing
struct CycleTiming {
  uint32_t cycleMs;
  uint32_t avDelayMs;
  uint32_t atriaContractMs;
  uint32_t atriaRelaxStartMs;
  uint32_t atriaRelaxMs;
  uint32_t ventContractMs;
  uint32_t ventHoldMs;
  uint32_t ventRelaxMs;
  uint32_t ventRelaxEndMs;
};
CycleTiming timing = {0};

// state
uint32_t cycleStartMs = 0;
uint32_t lastServoUpdateMs = 0;
uint32_t softStartBeginMs = 0;
uint32_t softStartEndMs = 0;

float bpmCurrent = 60.0f;
float potEMA = 0.0f;
uint32_t lastPotSampleMs = 0;

// remote control
bool remoteForced = false;           // MODE REMOTE
bool remoteActive = false;           // becomes true when we receive HR
uint32_t lastRemoteAt = 0;
float remoteTargetBPM = 60.0f;

// HR mode shaping
enum HrMode { MODE_NONE=0, MODE_BRADY, MODE_TACHY, MODE_AF };
HrMode hrMode = MODE_NONE;

// compute physiological timing
void computeTimings(float bpm) {
  bpm = clampf(bpm, BPM_MIN, BPM_MAX);
  uint32_t cycleMs = (uint32_t)(60000.0f / bpm);
  uint32_t diastolicMin = clampu32((uint32_t)(DIASTOLIC_MIN_FRAC * cycleMs),
                                   DIASTOLIC_MIN_MS_LO, DIASTOLIC_MIN_MS_HI);

  uint32_t avDelay = (uint32_t)clampf(0.14f * cycleMs, 100.0f, 170.0f);
  uint32_t atriaContract = (uint32_t)clampf(0.16f * cycleMs, 100.0f, 180.0f);
  if (atriaContract + ATRIA_VENT_GAP_MS > avDelay) {
    atriaContract = (avDelay > ATRIA_VENT_GAP_MS) ? (avDelay - ATRIA_VENT_GAP_MS) : (avDelay / 2);
    atriaContract = clampu32(atriaContract, 70, 180);
  }

  uint32_t ventTotal = (uint32_t)clampf(0.36f * cycleMs, 260.0f, 420.0f);
  uint32_t ventContract = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);
  uint32_t ventHold = (uint32_t)clampf(0.20f * ventTotal, 70.0f, 150.0f);
  uint32_t ventRelax = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);

  uint32_t vEnd = avDelay + ventContract + ventHold + ventRelax;
  if (cycleMs - vEnd < diastolicMin) {
    uint32_t need = diastolicMin - (cycleMs - vEnd);
    uint32_t shrinkable = ventContract + ventHold + ventRelax;
    if (need < shrinkable) {
      float s = (float)(shrinkable - need) / (float)shrinkable;
      ventContract = (uint32_t)(ventContract * s);
      ventHold = (uint32_t)(ventHold * s);
      ventRelax = (uint32_t)(ventRelax * s);
    }
    vEnd = avDelay + ventContract + ventHold + ventRelax;
  }

  uint32_t atriaRelaxStart = avDelay;
  uint32_t atriaRelax = (uint32_t)clampf(0.22f * cycleMs, 100.0f, 220.0f);

  timing = {cycleMs, avDelay, atriaContract, atriaRelaxStart,
             atriaRelax, ventContract, ventHold, ventRelax, vEnd};
}

float softstartScale(uint32_t nowMs) {
  if (nowMs <= softStartBeginMs) return 0.0f;
  if (nowMs >= softStartEndMs) return 1.0f;
  float x = (float)(nowMs - softStartBeginMs) / (float)(softStartEndMs - softStartBeginMs);
  return easeSmoothStep(x);
}

// read POT or REMOTE to determine target BPM
float targetBPMFromInputs(uint32_t nowMs) {
  // Auto fallback: if remote was active but stale, drop back to POT
  if (!remoteForced && remoteActive && (nowMs - lastRemoteAt > 2000)) {
    remoteActive = false;
  }

  if (remoteForced || remoteActive) {
    return clampf(remoteTargetBPM, BPM_MIN, BPM_MAX);
  }

  // POT mode
  if (nowMs - lastPotSampleMs >= POT_UPDATE_MS) {
    lastPotSampleMs = nowMs;
    int raw = analogRead(POT_PIN);
    if (potEMA <= 0.01f) potEMA = raw;
    potEMA = potEMA + POT_EMA_ALPHA * ((float)raw - potEMA);
  }
  float potBPM = BPM_MIN + (BPM_MAX - BPM_MIN) * (potEMA / 1023.0f);
  return clampf(potBPM, BPM_MIN, BPM_MAX);
}

void updateBPM(uint32_t nowMs) {
  float target = targetBPMFromInputs(nowMs);

  // Smooth approach; add small character based on HR mode
  float rate = 0.06f; // base smoothing per update (about 60ms tick)
  if (hrMode==MODE_BRADY)     rate = 0.04f;
  else if (hrMode==MODE_TACHY)rate = 0.08f;
  else if (hrMode==MODE_AF)   rate = 0.10f;

  bpmCurrent += (target - bpmCurrent) * rate;

  // Add tiny jitter character
  float jit = 0.0f;
  if (hrMode==MODE_NONE)      jit = 0.2f;
  else if (hrMode==MODE_BRADY)jit = 0.1f;
  else if (hrMode==MODE_TACHY)jit = 0.3f;
  else if (hrMode==MODE_AF)   jit = 1.0f;
  bpmCurrent += (float)random(-100,101) * 0.001f * jit;

  bpmCurrent = clampf(bpmCurrent, BPM_MIN, BPM_MAX);
}

// main heart motion
void updateHeart(uint32_t nowMs) {
  if (nowMs - cycleStartMs >= timing.cycleMs) {
    cycleStartMs = nowMs;
    computeTimings(bpmCurrent);
  }
  if (nowMs - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  lastServoUpdateMs = nowMs;
  uint32_t t = nowMs - cycleStartMs;
  float amp = softstartScale(nowMs);

  // Atria motion
  float a_up = easeInCubic(linProgress(t, 0, timing.atriaContractMs));
  float a_down = easeOutCubic(linProgress(t, timing.atriaRelaxStartMs, timing.atriaRelaxMs));
  float a_act = clampf(a_up * (1.0f - a_down), 0.0f, 1.0f) * amp;

  float aL = cal[0].relaxedAngleDeg + (cal[0].contractedAngleDeg - cal[0].relaxedAngleDeg) * a_act;
  float aR = cal[1].relaxedAngleDeg + (cal[1].contractedAngleDeg - cal[1].relaxedAngleDeg) * a_act;

  // Ventricle motion
  float v_relaxed = cal[2].relaxedAngleDeg;
  float v_contr = cal[2].contractedAngleDeg;
  uint32_t v_t0 = timing.avDelayMs;
  uint32_t v_t1 = v_t0 + timing.ventContractMs;
  uint32_t v_t2 = v_t1 + timing.ventHoldMs;
  uint32_t v_t3 = v_t2 + timing.ventRelaxMs;
  float ventAngle = v_relaxed;
  float v_up = easeInCubic(linProgress(t, v_t0, timing.ventContractMs));
  ventAngle = v_relaxed + (v_contr - v_relaxed) * v_up;
  if (t >= v_t1 && t < v_t2) ventAngle = v_contr;
  float v_down = easeOutQuint(linProgress(t, v_t2, timing.ventRelaxMs));
  if (t >= v_t2) ventAngle = v_contr + (v_relaxed - v_contr) * v_down;
  ventAngle = VENT_BASE_DEG + (ventAngle - VENT_BASE_DEG) * amp;
  ventAngle = clampf(ventAngle, VENT_SAFE_MIN_DEG, VENT_SAFE_MAX_DEG);

  // Write to servos
  servoAtriaL.write(aL);
  servoAtriaR.write(aR);
  servoVent.write(ventAngle);
}

// ---------- Serial CLI ----------
const size_t CMD_LEN = 64;
char sbuf[CMD_LEN];
uint8_t slen = 0;

void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  HR <bpm>            : set target BPM via serial"));
  Serial.println(F("  HRMODE none|brady|tachy|af"));
  Serial.println(F("  MODE POT|REMOTE     : force mode"));
  Serial.println(F("  STATUS | PING | HELP"));
}
void printStatus() {
  Serial.print(F("Mode: ")); Serial.println(remoteForced || remoteActive ? F("REMOTE") : F("POT"));
  Serial.print(F("BPM: ")); Serial.println(bpmCurrent,1);
}

void handleLine(char* line) {
  // trim leading spaces
  while(*line==' '||*line=='\t') line++;
  if(*line==0) return;

  // tokenise
  char tmp[CMD_LEN]; strncpy(tmp, line, CMD_LEN-1); tmp[CMD_LEN-1]=0;
  char* save=nullptr; char* tok=strtok_r(tmp," \t",&save);
  if(!tok) return;

  // uppercase first token
  for(char* p=tok; *p; ++p) *p=toupper(*p);

  if(!strcmp(tok,"PING")) {
    Serial.println(F("OK"));
  } else if(!strcmp(tok,"HELP")) {
    printHelp();
  } else if(!strcmp(tok,"STATUS")) {
    printStatus();
  } else if(!strcmp(tok,"MODE")) {
    char* arg=strtok_r(nullptr," \t",&save);
    if(!arg){ Serial.println(F("ERR: MODE needs POT or REMOTE")); return; }
    for(char* p=arg; *p; ++p) *p=toupper(*p);
    if(!strcmp(arg,"POT"))    { remoteForced=false; remoteActive=false; Serial.println(F("OK")); }
    else if(!strcmp(arg,"REMOTE")) { remoteForced=true;  remoteActive=true; Serial.println(F("OK")); }
    else Serial.println(F("ERR"));
  } else if(!strcmp(tok,"HR")) {
    char* a=strtok_r(nullptr," \t",&save);
    if(!a){ Serial.println(F("ERR")); return; }
    float b=(float)strtod(a,nullptr);
    if(b<30 || b>220){ Serial.println(F("ERR")); return; }
    remoteTargetBPM = b;
    remoteActive = true;
    lastRemoteAt = millis();
    Serial.println(F("OK"));
  } else if(!strcmp(tok,"HRMODE")) {
    char* a=strtok_r(nullptr," \t",&save);
    if(!a){ Serial.println(F("ERR")); return; }
    for(char* p=a; *p; ++p) *p=tolower(*p);
    if(!strcmp(a,"none")) hrMode=MODE_NONE;
    else if(!strcmp(a,"brady")||!strcmp(a,"bradycardia")) hrMode=MODE_BRADY;
    else if(!strcmp(a,"tachy")||!strcmp(a,"tachycardia")) hrMode=MODE_TACHY;
    else if(!strcmp(a,"af")||!strcmp(a,"afib")) hrMode=MODE_AF;
    else { Serial.println(F("ERR")); return; }
    Serial.println(F("OK"));
  } else {
    Serial.println(F("ERR"));
  }
}

void pollSerial() {
  while(Serial.available()>0) {
    char c=(char)Serial.read();
    if(c=='\r') continue;
    if(c=='\n'){
      sbuf[slen]=0;
      if(slen>0) handleLine(sbuf);
      slen=0;
    } else {
      if(slen<CMD_LEN-1) sbuf[slen++]=c;
      else slen=0;
    }
  }
}

// setup
void setup() {
  Serial.begin(115200);
  delay(POWER_ON_DELAY_MS);

  servoAtriaL.attach(SERVO_ATRIA_L_PIN);
  servoAtriaR.attach(SERVO_ATRIA_R_PIN);
  servoVent.attach(SERVO_VENT_PIN);

  potEMA = analogRead(POT_PIN);
  computeTimings(bpmCurrent);
  cycleStartMs = millis();
  softStartBeginMs = cycleStartMs;
  softStartEndMs = softStartBeginMs + SOFTSTART_RAMP_MS;

  servoAtriaL.write(cal[0].relaxedAngleDeg);
  servoAtriaR.write(cal[1].relaxedAngleDeg);
  servoVent.write(VENT_BASE_DEG);

  Serial.println(F("Nilheim Heart – REMOTE-SYNC ready. Type HELP."));
}

// loop
void loop() {
  uint32_t now = millis();
  pollSerial();
  updateBPM(now);
  updateHeart(now);
  delay(2);
}
