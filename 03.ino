/*
  Nilheim Mechatronics Servo Heart – biologically sequenced driver (rev C)
  Requires: Adafruit PWM Servo Driver Library
  https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library

  Changes in rev C:
  - Atria compute a shared activation and map through each servo's own calibration (fixes ch1 “no motion”)
  - PWM change deadband tightened to 1 for finer atrial updates
  - Keeps rev B improvements: soft-start, synchrony guards, diastolic reserve, non-blocking, EMA pot, slew-limited BPM
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ============ User configuration ============

// I2C address for PCA9685 (0x40 is default for most boards)
constexpr uint8_t PCA9685_ADDR = 0x40;

// Servo channels
constexpr uint8_t SERVO_ATRIA_L = 0;
constexpr uint8_t SERVO_ATRIA_R = 1;
constexpr uint8_t SERVO_VENT    = 2;

// Analog pin for BPM control
constexpr uint8_t POT_PIN = A0;

// PCA9685 frequency (60 Hz typical for analog servos)
constexpr float SERVO_PWM_HZ = 60.0f;

// Pot to BPM mapping and filtering
constexpr int    BPM_MIN = 45;
constexpr int    BPM_MAX = 140;
constexpr float  POT_EMA_ALPHA = 0.08f;  // 0..1; lower = smoother
constexpr uint32_t POT_UPDATE_MS = 60;   // resample pot every this many ms

// Motion smoothing: minimum time between command updates
constexpr uint32_t SERVO_UPDATE_MS = 15; // ~66 Hz command rate

// Soft-start (power and first beats)
constexpr uint32_t POWER_ON_DELAY_MS   = 1500;  // let supply settle
constexpr uint32_t SOFTSTART_RAMP_MS   = 1200;  // ease amplitude into first beats

// Physiologic guards
constexpr uint32_t ATRIA_VENT_GAP_MS   = 30;    // atria finish at least this before vent starts
constexpr float    DIASTOLIC_MIN_FRAC  = 0.18f; // keep ≥18% of cycle for diastole
constexpr uint32_t DIASTOLIC_MIN_MS_LO = 120;   // clamp floor
constexpr uint32_t DIASTOLIC_MIN_MS_HI = 400;   // clamp ceiling

// I2C update deadband (PWM counts 0..4095); avoid spam for tiny changes
constexpr uint16_t PWM_CHANGE_DEADBAND = 1;

// ============ Small helpers (non-templated to avoid Arduino prototype issues) ============
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static inline uint32_t clampu32(uint32_t v, uint32_t lo, uint32_t hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static inline uint16_t absDiffU16(uint16_t a, uint16_t b) {
  return (a > b) ? (uint16_t)(a - b) : (uint16_t)(b - a);
}

// Easing function: cubic smoothstep (no powf needed)
inline float easeSmoothStep(float x) {
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return x * x * (3.0f - 2.0f * x); // 3x^2 - 2x^3
}

// Per-servo calibration: set pulses (0..4095) and motion angles (deg 0..180)
struct ServoCal {
  uint16_t minPulse;           // pulse count at 0 deg
  uint16_t maxPulse;           // pulse count at 180 deg
  float    relaxedAngleDeg;    // resting position
  float    contractedAngleDeg; // peak contraction position
};

// Defaults are conservative; tune to your mechanics.
ServoCal cal[3] = {
  // Atria L
  { 140, 520,  25.0f,  55.0f },
  // Atria R
  { 140, 520,  25.0f,  55.0f },
  // Ventricle
  { 140, 520,  35.0f, 115.0f }
};

// ============ Internals ============

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

// Cache last pulse to avoid redundant I2C updates
uint16_t lastPulse[3] = { 0xFFFF, 0xFFFF, 0xFFFF };

// Pot filtering
uint16_t potRaw = 0;
float    potEMA = 0.0f;
uint32_t lastPotSampleMs = 0;

// BPM smoothing: slew-rate limit BPM changes to avoid sudden shifts
float    bpmCurrent = 60.0f;     // Start at a sane value
constexpr float BPM_SLEW_PER_SEC = 60.0f; // max BPM change per second

// Timing the heart cycle
uint32_t cycleStartMs = 0;
uint32_t lastServoUpdateMs = 0;

// Soft-start window
uint32_t softStartBeginMs = 0;
uint32_t softStartEndMs   = 0;

// Precomputed phase durations for the current cycle (in ms)
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

// Convert angle (deg) to pulse count for a specific servo
uint16_t angleToPulse(uint8_t servoIdx, float angleDeg) {
  angleDeg = clampf(angleDeg, 0.0f, 180.0f);
  const ServoCal &c = cal[servoIdx];
  float t = angleDeg / 180.0f;
  float pulse = c.minPulse + t * (float)(c.maxPulse - c.minPulse);
  int p = clampi((int)pulse, 0, 4095);
  return (uint16_t)p;
}

// Write angle to a servo with change suppression + deadband
void writeServoAngle(uint8_t servoIdx, float angleDeg) {
  uint16_t pulse = angleToPulse(servoIdx, angleDeg);
  if (lastPulse[servoIdx] == 0xFFFF || absDiffU16(lastPulse[servoIdx], pulse) >= PWM_CHANGE_DEADBAND) {
    pwm.setPWM(servoIdx, 0, pulse);
    lastPulse[servoIdx] = pulse;
  }
}

// Compute heart timings from BPM with physiologic guards
void computeTimings(float bpm) {
  bpm = clampf(bpm, (float)BPM_MIN, (float)BPM_MAX);
  uint32_t cycleMs = (uint32_t)(60000.0f / bpm);

  // Keep a real diastolic reserve
  uint32_t diastolicMin = clampu32((uint32_t)(DIASTOLIC_MIN_FRAC * cycleMs),
                                   DIASTOLIC_MIN_MS_LO, DIASTOLIC_MIN_MS_HI);

  // AV delay ~100–170 ms (scaled)
  uint32_t avDelay = (uint32_t)clampf(0.14f * cycleMs, 100.0f, 170.0f);

  // Atrial contraction 100–180 ms, must finish before vent starts
  uint32_t atriaContract = (uint32_t)clampf(0.16f * cycleMs, 100.0f, 180.0f);
  if (atriaContract + ATRIA_VENT_GAP_MS > avDelay) {
    atriaContract = (avDelay > ATRIA_VENT_GAP_MS) ? (avDelay - ATRIA_VENT_GAP_MS) : (avDelay / 2);
    atriaContract = clampu32(atriaContract, 70, 180);
  }

  // Ventricular systole ~36% of cycle (then split)
  uint32_t ventTotal = (uint32_t)clampf(0.36f * cycleMs, 260.0f, 420.0f);
  uint32_t ventContract = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);
  uint32_t ventHold     = (uint32_t)clampf(0.20f * ventTotal,  70.0f, 150.0f);
  uint32_t ventRelax    = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);

  // Ensure diastolic reserve
  uint32_t vEnd = avDelay + ventContract + ventHold + ventRelax;
  if (cycleMs - vEnd < diastolicMin) {
    uint32_t need = diastolicMin - (cycleMs - vEnd);
    uint32_t shrinkable = ventContract + ventHold + ventRelax;
    if (need < shrinkable) {
      float s = (float)(shrinkable - need) / (float)shrinkable;
      ventContract = clampu32((uint32_t)(ventContract * s), 100, 220);
      ventHold     = clampu32((uint32_t)(ventHold     * s),  60, 140);
      ventRelax    = clampu32((uint32_t)(ventRelax    * s), 100, 220);
    } else {
      ventContract = 100; ventHold = 60; ventRelax = 100;
    }
    vEnd = avDelay + ventContract + ventHold + ventRelax;
  }

  // Atria begin relaxing as the ventricle starts (mimics AV valve closure timing)
  uint32_t atriaRelaxStart = avDelay;
  uint32_t atriaRelax      = (uint32_t)clampf(0.22f * cycleMs, 100.0f, 220.0f);

  timing.cycleMs           = cycleMs;
  timing.avDelayMs         = avDelay;
  timing.atriaContractMs   = atriaContract;
  timing.atriaRelaxStartMs = atriaRelaxStart;
  timing.atriaRelaxMs      = atriaRelax;
  timing.ventContractMs    = ventContract;
  timing.ventHoldMs        = ventHold;
  timing.ventRelaxMs       = ventRelax;
  timing.ventRelaxEndMs    = vEnd;
}

// Eased progress for a motion that starts at t0 and lasts durMs (0..1)
float easedProgress(uint32_t nowInCycleMs, uint32_t t0, uint32_t durMs) {
  if (nowInCycleMs <= t0) return 0.0f;
  uint32_t dt = nowInCycleMs - t0;
  if (dt >= durMs) return 1.0f;
  float x = (float)dt / (float)durMs;
  return easeSmoothStep(x);
}

// Soft-start amplitude (ramps 0→1 during first ~1.2s)
float softstartScale(uint32_t nowMs) {
  if (nowMs <= softStartBeginMs) return 0.0f;
  if (nowMs >= softStartEndMs)   return 1.0f;
  float x = (float)(nowMs - softStartBeginMs) / (float)(softStartEndMs - softStartBeginMs);
  return easeSmoothStep(x);
}

// Update pot and compute target BPM smoothly
void updateBPM(uint32_t nowMs) {
  if (nowMs - lastPotSampleMs >= POT_UPDATE_MS) {
    lastPotSampleMs = nowMs;
    uint16_t raw = analogRead(POT_PIN); // 0..1023
    potRaw = raw;
    // initialize EMA on first run
    if (potEMA <= 0.01f) potEMA = (float)raw;
    potEMA = potEMA + POT_EMA_ALPHA * ((float)raw - potEMA);

    // Map filtered pot to target BPM
    float targetBPM = BPM_MIN + (BPM_MAX - BPM_MIN) * (potEMA / 1023.0f);

    // Slew-rate limit BPM changes to avoid sudden shifts
    static uint32_t lastMs = nowMs;
    float dtSec = (nowMs - lastMs) / 1000.0f;
    lastMs = nowMs;
    float maxDelta = BPM_SLEW_PER_SEC * dtSec;
    float delta = targetBPM - bpmCurrent;
    if (delta >  maxDelta) delta =  maxDelta;
    if (delta < -maxDelta) delta = -maxDelta;
    bpmCurrent += delta;
  }
}

// Drives servos according to current time within the cycle
void updateHeart(uint32_t nowMs) {
  // Start a new cycle if we’re past the end
  if (nowMs - cycleStartMs >= timing.cycleMs) {
    cycleStartMs = nowMs;
    // recompute timings at the start of the cycle, in case BPM changed
    computeTimings(bpmCurrent);
  }

  // Only update servos at a limited rate
  if (nowMs - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  lastServoUpdateMs = nowMs;

  uint32_t t = nowMs - cycleStartMs;

  // Global amplitude scale for first ~1.2s after start
  float amp = softstartScale(nowMs);

  // ---------- ATRIA ----------
  // Shared activation: ramps up during contraction, then back down during relax
  float a_prog_contract = easedProgress(t, 0,                      timing.atriaContractMs);
  float a_prog_relax    = easedProgress(t, timing.atriaRelaxStartMs, timing.atriaRelaxMs);

  float a_act = a_prog_contract + (0.0f - a_prog_contract) * a_prog_relax; // up then down
  a_act = clampf(a_act, 0.0f, 1.0f);
  a_act *= amp; // soft-start amplitude

  // Left atrium (own calibration)
  float aL = cal[SERVO_ATRIA_L].relaxedAngleDeg +
             (cal[SERVO_ATRIA_L].contractedAngleDeg - cal[SERVO_ATRIA_L].relaxedAngleDeg) * a_act;

  // Right atrium (own calibration)
  float aR = cal[SERVO_ATRIA_R].relaxedAngleDeg +
             (cal[SERVO_ATRIA_R].contractedAngleDeg - cal[SERVO_ATRIA_R].relaxedAngleDeg) * a_act;

  // ---------- VENTRICLE ----------
  float v_relaxed = cal[SERVO_VENT].relaxedAngleDeg;
  float v_contr   = cal[SERVO_VENT].contractedAngleDeg;

  uint32_t v_t0 = timing.avDelayMs;
  uint32_t v_t1 = v_t0 + timing.ventContractMs;
  uint32_t v_t2 = v_t1 + timing.ventHoldMs;
  uint32_t v_t3 = v_t2 + timing.ventRelaxMs; // equals timing.ventRelaxEndMs

  float ventAngle = v_relaxed;

  // Contract phase
  float v_prog_contract = easedProgress(t, v_t0, timing.ventContractMs);
  ventAngle = v_relaxed + (v_contr - v_relaxed) * v_prog_contract;

  // Hold phase: keep at contracted
  if (t >= v_t1 && t < v_t2) {
    ventAngle = v_contr;
  }

  // Relax phase
  float v_prog_relax = easedProgress(t, v_t2, timing.ventRelaxMs);
  if (t >= v_t2) {
    ventAngle = v_contr + (v_relaxed - v_contr) * v_prog_relax;
  }

  // Apply soft-start amplitude
  ventAngle = v_relaxed + (ventAngle - v_relaxed) * amp;

  // ---------- WRITE ----------
  writeServoAngle(SERVO_ATRIA_L, aL);
  writeServoAngle(SERVO_ATRIA_R, aR);
  writeServoAngle(SERVO_VENT,    ventAngle);
}

void setup() {
  pinMode(POT_PIN, INPUT);

  Serial.begin(115200);
  delay(50);
  Serial.println("Nilheim Heart – biologically sequenced servo driver (rev C)");

  Wire.begin();
  // Safer I2C speed: many boards support 400 kHz; ignore if not.
  #if defined(ARDUINO_ARCH_AVR) || defined(ARDUINO_ARCH_SAMD) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    Wire.setClock(400000);
  #endif

  pwm.begin();
  pwm.setPWMFreq(SERVO_PWM_HZ);
  delay(10);

  // Give power rails time to settle before holding any position
  delay(POWER_ON_DELAY_MS);

  // Initialize EMA
  potRaw = analogRead(POT_PIN);
  potEMA = (float)potRaw;

  // Initialize timings
  computeTimings(bpmCurrent);
  cycleStartMs = millis();

  // Soft-start window
  softStartBeginMs = cycleStartMs;                 // ramp begins with first cycle
  softStartEndMs   = softStartBeginMs + SOFTSTART_RAMP_MS;

  // Initialize servos at relaxed positions
  writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
  writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
  writeServoAngle(SERVO_VENT,    cal[SERVO_VENT].relaxedAngleDeg);
}

void loop() {
  uint32_t now = millis();
  updateBPM(now);
  updateHeart(now);

  // Small idle to keep loop cooperative
  delay(2);
}
