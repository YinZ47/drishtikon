/*
  Nilheim Mechatronics Servo Heart – biologically sequenced driver
  Requires: Adafruit PWM Servo Driver Library
  https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library

  Hardware:
  - PCA9685 on I2C (default addr 0x40)
  - Servos on channels 0, 1 (atria L/R), 2 (ventricle)
  - Potentiometer on A0 to control heart rate (BPM)

  Key improvements:
  - Proper cardiac sequence: atrial systole -> AV delay -> ventricular systole (contract/hold/relax) -> diastole/fill
  - millis()-based timing, no blocking loops; smooth easing curves for motion
  - Pot input filtered (EMA) to avoid jitter; BPM transitions are gradual
  - Per-servo calibration for pulse endpoints and motion angles
  - I2C updates only when output changes to reduce bus jitter

  Calibration workflow (do this once):
  1) Set PWM endpoints for each servo (minPulse, maxPulse). Start with 500–2500 µs equivalents (~110–535 counts at 60 Hz) but use the defaults below as a typical starting point.
  2) Tune relaxedAngle/contractedAngle (in degrees, 0–180) so mechanisms don’t bind. Start small and increase until motion looks right.
  3) Use BPM range that suits your mechanics. Default 45–140 BPM.

  Notes:
  - If your servos buzz or stutter, slightly reduce travel or slow contraction times.
  - If you see any binding at extremes, back off angles or adjust min/max pulse.
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

// Easing function: use smoothstep-like easing for servo moves
inline float easeInOutCubic(float x) {
  // Clamp first
  if (x <= 0.0f) return 0.0f;
  if (x >= 1.0f) return 1.0f;
  return (x < 0.5f) ? 4.0f * x * x * x
                    : 1.0f - powf(-2.0f * x + 2.0f, 3.0f) / 2.0f;
}

// Per-servo calibration: set pulses (0..4095) and motion angles (deg 0..180)
struct ServoCal {
  uint16_t minPulse;        // pulse count at 0 deg
  uint16_t maxPulse;        // pulse count at 180 deg
  float    relaxedAngleDeg; // resting position
  float    contractedAngleDeg; // peak contraction position
};

// Defaults are conservative; tune to your mechanics.
// These pulse counts correspond roughly to ~500–2500 µs at 60 Hz for many servos.
// If your servo endpoints differ, change minPulse/maxPulse per servo.
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

// BPM smoothing: we’ll slew-rate limit BPM changes to avoid sudden shifts
float    bpmCurrent = 60.0f;     // Start at a sane value
constexpr float BPM_SLEW_PER_SEC = 60.0f; // max BPM change per second

// Timing the heart cycle
uint32_t cycleStartMs = 0;
uint32_t lastServoUpdateMs = 0;

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

// Utility: clamp
template <typename T>
T clamp(T v, T lo, T hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Convert angle (deg) to pulse count for a specific servo
uint16_t angleToPulse(uint8_t servoIdx, float angleDeg) {
  angleDeg = clamp(angleDeg, 0.0f, 180.0f);
  const ServoCal &c = cal[servoIdx];
  float t = angleDeg / 180.0f;
  float pulse = c.minPulse + t * (float)(c.maxPulse - c.minPulse);
  uint16_t p = (uint16_t)clamp((int)pulse, 0, 4095);
  return p;
}

// Write angle to a servo with change suppression
void writeServoAngle(uint8_t servoIdx, float angleDeg) {
  uint16_t pulse = angleToPulse(servoIdx, angleDeg);
  if (lastPulse[servoIdx] != pulse) {
    pwm.setPWM(servoIdx, 0, pulse);
    lastPulse[servoIdx] = pulse;
  }
}

// Compute heart timings from BPM
void computeTimings(float bpm) {
  bpm = clamp(bpm, (float)BPM_MIN, (float)BPM_MAX);
  uint32_t cycleMs = (uint32_t)(60000.0f / bpm);

  // Physiological-inspired allocations with safe clamps for servos
  // AV delay ~80–180 ms, scaled with cycle
  uint32_t avDelay = (uint32_t)clamp(0.12f * cycleMs, 80.0f, 180.0f);

  // Atrial contraction 90–200 ms
  uint32_t atriaContract = (uint32_t)clamp(0.18f * cycleMs, 90.0f, 200.0f);

  // Ventricular systole total ~ 0.3–0.4 of cycle, but clamped
  uint32_t ventSystole = (uint32_t)clamp(0.35f * cycleMs, 260.0f, 380.0f);

  // Break ventricular systole into segments
  uint32_t ventContract = (uint32_t)clamp(0.40f * ventSystole, 120.0f, 220.0f);
  uint32_t ventHold     = (uint32_t)clamp(0.20f * ventSystole,  70.0f, 150.0f);
  uint32_t ventRelax    = (uint32_t)clamp(0.40f * ventSystole, 120.0f, 220.0f);

  // Atria relax starts a bit after vent starts contracting
  uint32_t atriaRelaxStart = avDelay + clamp((uint32_t)60, (uint32_t)40, (uint32_t)100);
  uint32_t atriaRelax      = (uint32_t)clamp(0.22f * cycleMs, 110.0f, 220.0f);

  timing.cycleMs           = cycleMs;
  timing.avDelayMs         = avDelay;
  timing.atriaContractMs   = atriaContract;
  timing.atriaRelaxStartMs = atriaRelaxStart;
  timing.atriaRelaxMs      = atriaRelax;
  timing.ventContractMs    = ventContract;
  timing.ventHoldMs        = ventHold;
  timing.ventRelaxMs       = ventRelax;
  timing.ventRelaxEndMs    = avDelay + ventContract + ventHold + ventRelax;
}

// Get eased position for a motion that starts at t0 and lasts durMs
// Returns 0..1 eased progress; 0 when before start, 1 when after end.
float easedProgress(uint32_t nowMs, uint32_t t0, uint32_t durMs) {
  if (nowMs <= t0) return 0.0f;
  uint32_t dt = nowMs - t0;
  if (dt >= durMs) return 1.0f;
  float x = (float)dt / (float)durMs;
  return easeInOutCubic(x);
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
    if (delta > maxDelta) delta = maxDelta;
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

  // ATRIA: contract from relaxed to contracted [0 .. atriaContractMs]
  float a_relaxed = cal[SERVO_ATRIA_L].relaxedAngleDeg;   // same for L/R
  float a_contr   = cal[SERVO_ATRIA_L].contractedAngleDeg;

  float a_prog_contract = easedProgress(t, 0, timing.atriaContractMs);
  float atriaAngle = a_relaxed + (a_contr - a_relaxed) * a_prog_contract;

  // After atriaRelaxStartMs, relax back over atriaRelaxMs
  float a_prog_relax = easedProgress(t, timing.atriaRelaxStartMs, timing.atriaRelaxMs);
  // Blend back to relaxed
  atriaAngle = atriaAngle + (a_relaxed - atriaAngle) * a_prog_relax;

  // VENTRICLE:
  float v_relaxed = cal[SERVO_VENT].relaxedAngleDeg;
  float v_contr   = cal[SERVO_VENT].contractedAngleDeg;

  uint32_t v_t0 = timing.avDelayMs;
  uint32_t v_t1 = v_t0 + timing.ventContractMs;
  uint32_t v_t2 = v_t1 + timing.ventHoldMs;
  uint32_t v_t3 = v_t2 + timing.ventRelaxMs; // should equal timing.ventRelaxEndMs

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

  // Write outputs (both atria get same angle)
  writeServoAngle(SERVO_ATRIA_L, atriaAngle);
  writeServoAngle(SERVO_ATRIA_R, atriaAngle);
  writeServoAngle(SERVO_VENT,    ventAngle);
}

void setup() {
  pinMode(POT_PIN, INPUT);

  Serial.begin(115200);
  delay(50);
  Serial.println("Nilheim Heart – biologically sequenced servo driver");

  pwm.begin();
  // Faster I2C helps reduce jitter (PCA9685 supports up to 1 MHz; many Arduinos do 400 kHz)
  Wire.setClock(400000);
  pwm.setPWMFreq(SERVO_PWM_HZ);
  delay(10);

  // Initialize EMA
  potRaw = analogRead(POT_PIN);
  potEMA = (float)potRaw;

  // Initialize timings
  computeTimings(bpmCurrent);
  cycleStartMs = millis();

  // Initialize servos at relaxed positions
  writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
  writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
  writeServoAngle(SERVO_VENT,    cal[SERVO_VENT].relaxedAngleDeg);
}

void loop() {
  uint32_t now = millis();
  updateBPM(now);
  updateHeart(now);

  // Optional: very small idle to keep loop cooperative
  delay(2);
}
