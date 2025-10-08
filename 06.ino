/*
  Nilheim Mechatronics Servo Heart – biologically sequenced driver (rev D′ with ventricle calibration)

  Same as your rev D (merged + protected), but with ventricle calibrated like your direct-pin test:
    - Baseline (relaxed) = 90°
    - Safe motion limits = 70° .. 130° (hard clamp)
    - Contracted peak    = 130°
  Atria remain as before. PCA9685 drives all three channels: ch0 LA, ch1 RA, ch2 Ventricle.
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>   // for isfinite, fabsf

// ============ User configuration ============
// PCA9685
constexpr uint8_t PCA9685_ADDR = 0x40;
constexpr float   SERVO_PWM_HZ = 60.0f;

// Servo channels (PCA9685)
constexpr uint8_t SERVO_ATRIA_L = 0;
constexpr uint8_t SERVO_ATRIA_R = 1;
constexpr uint8_t SERVO_VENT    = 2;

// Potentiometer
constexpr uint8_t POT_PIN = A0;

// Pot → BPM mapping & filtering
constexpr int     BPM_MIN = 45;
constexpr int     BPM_MAX = 140;
constexpr float   POT_EMA_ALPHA = 0.08f;
constexpr uint32_t POT_UPDATE_MS = 60;

// Command cadence
constexpr uint32_t SERVO_UPDATE_MS = 15; // ~66 Hz

// Soft-start
constexpr uint32_t POWER_ON_DELAY_MS = 1500; // supply settle
constexpr uint32_t SOFTSTART_RAMP_MS = 1200; // amplitude ramp

// Physiology guards
constexpr uint32_t ATRIA_VENT_GAP_MS   = 30;     // atria finish at least this before vent starts
constexpr float    DIASTOLIC_MIN_FRAC  = 0.18f;  // ≥18% diastole
constexpr uint32_t DIASTOLIC_MIN_MS_LO = 120;
constexpr uint32_t DIASTOLIC_MIN_MS_HI = 400;

// I2C update suppression
constexpr uint16_t PWM_CHANGE_DEADBAND = 1;      // finer so atrial micro-steps pass

// Atria motion safety / feel
constexpr float    MAX_ANG_VEL_DEG_PER_SEC = 120.0f; // clamp overly fast moves
constexpr float    MIN_ANG_VEL_DEG_PER_SEC = 20.0f;  // ensure tiny moves don't stall
constexpr uint16_t MAX_PULSE_CHANGE_PER_UPDATE = 40; // cap raw 0..4095 change per write

// Safety: pot & loop
constexpr uint32_t EMERGENCY_COOLDOWN_MS = 3000;
constexpr float    POT_SPIKE_THRESHOLD = 200.0f;     // sudden raw jump triggers stop
constexpr uint32_t MAX_LOOP_JITTER_MS  = 200;        // watchdog

// -------- VENTRICLE CALIBRATION from your direct-pin test --------
constexpr float VENT_BASE_DEG     = 90.0f;   // baseline
constexpr float VENT_SAFE_MIN_DEG = 70.0f;   // absolute lower limit
constexpr float VENT_SAFE_MAX_DEG = 130.0f;  // absolute upper limit
constexpr float VENT_CONTRACT_DEG = 130.0f;  // peak during systole

// ============ Helpers ============
static inline float clampf(float v, float lo, float hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }
static inline int   clampi(int v, int lo, int hi)       { if (v < lo) return lo; if (v > hi) return hi; return v; }
static inline uint32_t clampu32(uint32_t v, uint32_t lo, uint32_t hi) { if (v < lo) return lo; if (v > hi) return hi; return v; }
static inline uint16_t absDiffU16(uint16_t a, uint16_t b) { return (a > b) ? (uint16_t)(a - b) : (uint16_t)(b - a); }
inline float easeSmoothStep(float x) { if (x <= 0.0f) return 0.0f; if (x >= 1.0f) return 1.0f; return x*x*(3.0f - 2.0f*x); }

// Per-servo calibration (0..4095 pulses map to 0..180°), atria as before, vent uses new angles
struct ServoCal {
  uint16_t minPulse;           // pulse count at 0 deg
  uint16_t maxPulse;           // pulse count at 180 deg
  float    relaxedAngleDeg;    // resting position
  float    contractedAngleDeg; // peak contraction position
};

ServoCal cal[3] = {
  // Atria L
  { 140, 520,  25.0f,  55.0f },
  // Atria R
  { 140, 520,  25.0f,  55.0f },
  // Ventricle (calibrated to your tester: base 90°, peak 130°; safe clamp 70..130 is enforced below)
  { 140, 520,  VENT_BASE_DEG,  VENT_CONTRACT_DEG }
};

// ============ Internals ============
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

uint16_t lastPulse[3] = { 0xFFFF, 0xFFFF, 0xFFFF };
float    lastAngleDeg[3] = { -999.0f, -999.0f, -999.0f };
uint32_t lastAngleTimeMs[3] = {0,0,0};

uint16_t potRaw = 0;
float    potEMA = 0.0f;
uint32_t lastPotSampleMs = 0;
uint16_t lastPotRaw = 0;

float    bpmCurrent = 60.0f;
constexpr float BPM_SLEW_PER_SEC = 60.0f;

uint32_t cycleStartMs = 0;
uint32_t lastServoUpdateMs = 0;

uint32_t softStartBeginMs = 0;
uint32_t softStartEndMs   = 0;

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

// Safety flags
bool     emergencyStopActive = false;
uint32_t emergencyStopSince  = 0;
uint32_t lastLoopMs          = 0;

// ============ Conversions & safe write ============
uint16_t angleToPulse(uint8_t servoIdx, float angleDeg) {
  angleDeg = clampf(angleDeg, 0.0f, 180.0f);
  const ServoCal &c = cal[servoIdx];
  float t = angleDeg / 180.0f;
  float pulse = c.minPulse + t * (float)(c.maxPulse - c.minPulse);
  int p = clampi((int)pulse, 0, 4095);
  return (uint16_t)p;
}

// Rate-limited, deadbanded, and pulse-cap write (+ ventricle hard safety window)
void writeServoAngle(uint8_t servoIdx, float angleDeg) {
  const ServoCal &c = cal[servoIdx];

  // Default safety: small margin around calibrated range
  float minA = c.relaxedAngleDeg - 10.0f;
  float maxA = c.contractedAngleDeg + 10.0f;

  // For the ventricle, enforce your absolute safe window 70..130 exactly
  if (servoIdx == SERVO_VENT) {
    minA = VENT_SAFE_MIN_DEG;
    maxA = VENT_SAFE_MAX_DEG;
  }

  angleDeg = clampf(angleDeg, minA, maxA);

  uint32_t now = millis();

  // Enforce angular velocity limits (for atria and ventricle alike; gentle)
  if (lastAngleDeg[servoIdx] > -900.0f) {
    uint32_t dtMs = now - lastAngleTimeMs[servoIdx];
    if (dtMs > 0) {
      float dtSec = dtMs / 1000.0f;
      float desiredDelta = angleDeg - lastAngleDeg[servoIdx];
      float ad = fabsf(desiredDelta);

      float maxDeltaDeg = MAX_ANG_VEL_DEG_PER_SEC * dtSec;
      float minDeltaDeg = MIN_ANG_VEL_DEG_PER_SEC * dtSec;

      if (ad > maxDeltaDeg) {
        angleDeg = lastAngleDeg[servoIdx] + (desiredDelta > 0 ? maxDeltaDeg : -maxDeltaDeg);
      } else if (ad > 0.0f && ad < minDeltaDeg) {
        float step = (ad < minDeltaDeg) ? ad : minDeltaDeg;
        angleDeg = lastAngleDeg[servoIdx] + (desiredDelta > 0 ? step : -step);
      }
    }
  }

  uint16_t pulse = angleToPulse(servoIdx, angleDeg);

  // Cap raw pulse jump per update & apply deadband
  if (lastPulse[servoIdx] != 0xFFFF) {
    int32_t diff = (int32_t)pulse - (int32_t)lastPulse[servoIdx];
    int32_t adiff = (diff >= 0) ? diff : -diff;
    if (adiff > (int32_t)MAX_PULSE_CHANGE_PER_UPDATE) {
      pulse = (uint16_t)((int32_t)lastPulse[servoIdx] + (diff > 0 ? (int32_t)MAX_PULSE_CHANGE_PER_UPDATE
                                                                  : -(int32_t)MAX_PULSE_CHANGE_PER_UPDATE));
    }
    if (absDiffU16(lastPulse[servoIdx], pulse) < PWM_CHANGE_DEADBAND) {
      return; // tiny change—skip I2C write
    }
  }

  pwm.setPWM(servoIdx, 0, pulse);
  lastPulse[servoIdx]      = pulse;
  lastAngleDeg[servoIdx]   = angleDeg;
  lastAngleTimeMs[servoIdx]= now;
}

// ============ Timing / physiology ============
void computeTimings(float bpm) {
  bpm = clampf(bpm, (float)BPM_MIN, (float)BPM_MAX);
  uint32_t cycleMs = (uint32_t)(60000.0f / bpm);

  uint32_t diastolicMin = clampu32((uint32_t)(DIASTOLIC_MIN_FRAC * cycleMs),
                                   DIASTOLIC_MIN_MS_LO, DIASTOLIC_MIN_MS_HI);

  // AV delay and atria contraction (ensure atria finish before AV)
  uint32_t avDelay = (uint32_t)clampf(0.14f * cycleMs, 100.0f, 170.0f);
  uint32_t atriaContract = (uint32_t)clampf(0.16f * cycleMs, 100.0f, 180.0f);
  if (atriaContract + ATRIA_VENT_GAP_MS > avDelay) {
    atriaContract = (avDelay > ATRIA_VENT_GAP_MS) ? (avDelay - ATRIA_VENT_GAP_MS) : (avDelay / 2);
    atriaContract = clampu32(atriaContract, 70, 180);
  }

  // Ventricular systole split
  uint32_t ventTotal   = (uint32_t)clampf(0.36f * cycleMs, 260.0f, 420.0f);
  uint32_t ventContract= (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);
  uint32_t ventHold    = (uint32_t)clampf(0.20f * ventTotal,  70.0f, 150.0f);
  uint32_t ventRelax   = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);

  uint32_t vEnd = avDelay + ventContract + ventHold + ventRelax;
  // Enforce diastolic reserve
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

  uint32_t atriaRelaxStart = avDelay; // begin relaxing as the ventricle starts
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

float easedProgress(uint32_t nowInCycleMs, uint32_t t0, uint32_t durMs) {
  if (nowInCycleMs <= t0) return 0.0f;
  uint32_t dt = nowInCycleMs - t0;
  if (dt >= durMs) return 1.0f;
  float x = (float)dt / (float)durMs;
  return easeSmoothStep(x);
}

float softstartScale(uint32_t nowMs) {
  if (nowMs <= softStartBeginMs) return 0.0f;
  if (nowMs >= softStartEndMs)   return 1.0f;
  float x = (float)(nowMs - softStartBeginMs) / (float)(softStartEndMs - softStartBeginMs);
  return easeSmoothStep(x);
}

// ============ Safety helpers ============
void enterEmergencyStop(const char *reason) {
  if (!emergencyStopActive) {
    emergencyStopActive = true;
    emergencyStopSince = millis();
    Serial.print("EMERGENCY STOP: ");
    Serial.println(reason);
    // Move toward relaxed (atria) and baseline (ventricle)
    writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
    writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
    writeServoAngle(SERVO_VENT,    VENT_BASE_DEG);
  }
}

void clearEmergencyIfSafe() {
  if (!emergencyStopActive) return;
  if (millis() - emergencyStopSince < EMERGENCY_COOLDOWN_MS) return;

  // simple stability check on pot
  if (abs((int)potRaw - (int)lastPotRaw) < 10) {
    emergencyStopActive = false;
    Serial.println("EMERGENCY CLEARED: pot stable and cooldown elapsed.");
  } else {
    lastPotRaw = potRaw;
  }
}

// ============ Input / BPM update ============
void updateBPM(uint32_t nowMs) {
  if (nowMs - lastPotSampleMs >= POT_UPDATE_MS) {
    lastPotSampleMs = nowMs;

    uint16_t raw = analogRead(POT_PIN); // 0..1023
    potRaw = raw;
    if (potEMA <= 0.01f) potEMA = (float)raw;
    potEMA = potEMA + POT_EMA_ALPHA * ((float)raw - potEMA);

    // Spike detection
    if (abs((int)raw - (int)lastPotRaw) > (int)POT_SPIKE_THRESHOLD) {
      enterEmergencyStop("pot spike detected");
    }
    lastPotRaw = raw;

    float targetBPM = BPM_MIN + (BPM_MAX - BPM_MIN) * (potEMA / 1023.0f);

    // Slew-limit BPM
    static uint32_t lastMs = nowMs;
    float dtSec = (nowMs - lastMs) / 1000.0f;
    if (dtSec < 0.0001f) dtSec = 0.0001f;
    lastMs = nowMs;

    float maxDelta = BPM_SLEW_PER_SEC * dtSec;
    float delta = targetBPM - bpmCurrent;
    if (delta >  maxDelta) delta =  maxDelta;
    if (delta < -maxDelta) delta = -maxDelta;
    bpmCurrent += delta;

    // Sanity
    if (!isfinite(potEMA) || potEMA < -100.0f || potEMA > 5000.0f) {
      enterEmergencyStop("pot EMA invalid");
    }
  }
}

// ============ Heart update ============
void updateHeart(uint32_t nowMs) {
  // Watchdog for long loop stalls
  if (lastLoopMs != 0 && nowMs - lastLoopMs > MAX_LOOP_JITTER_MS) {
    enterEmergencyStop("loop jitter too large");
  }
  lastLoopMs = nowMs;

  if (emergencyStopActive) {
    clearEmergencyIfSafe();
    return;
  }

  // New cycle boundary
  if (nowMs - cycleStartMs >= timing.cycleMs) {
    cycleStartMs = nowMs;
    computeTimings(bpmCurrent);
  }

  // Throttle servo commands
  if (nowMs - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  uint32_t dtServoUpdate = nowMs - lastServoUpdateMs;
  (void)dtServoUpdate; // kept for future guards
  lastServoUpdateMs = nowMs;

  uint32_t t = nowMs - cycleStartMs;
  float amp = softstartScale(nowMs);

  // ---------- ATRIA (shared activation → per-servo mapping) ----------
  float a_prog_contract = easedProgress(t, 0,                      timing.atriaContractMs);
  float a_prog_relax    = easedProgress(t, timing.atriaRelaxStartMs, timing.atriaRelaxMs);
  // activation rises (0→1) during contract, then falls during relax
  float a_act = a_prog_contract + (0.0f - a_prog_contract) * a_prog_relax;
  a_act = clampf(a_act, 0.0f, 1.0f) * amp;

  float aL = cal[SERVO_ATRIA_L].relaxedAngleDeg +
             (cal[SERVO_ATRIA_L].contractedAngleDeg - cal[SERVO_ATRIA_L].relaxedAngleDeg) * a_act;

  float aR = cal[SERVO_ATRIA_R].relaxedAngleDeg +
             (cal[SERVO_ATRIA_R].contractedAngleDeg - cal[SERVO_ATRIA_R].relaxedAngleDeg) * a_act;

  // ---------- VENTRICLE (PCA9685, calibrated to 90° base → 130° peak, clamped 70..130) ----------
  float v_relaxed = cal[SERVO_VENT].relaxedAngleDeg;   // 90°
  float v_contr   = cal[SERVO_VENT].contractedAngleDeg; // 130°

  uint32_t v_t0 = timing.avDelayMs;
  uint32_t v_t1 = v_t0 + timing.ventContractMs;
  uint32_t v_t2 = v_t1 + timing.ventHoldMs;
  uint32_t v_t3 = v_t2 + timing.ventRelaxMs; (void)v_t3;

  float ventAngle = v_relaxed;

  // Contract phase: 90° → 130°
  float v_prog_contract = easedProgress(t, v_t0, timing.ventContractMs);
  ventAngle = v_relaxed + (v_contr - v_relaxed) * v_prog_contract;

  // Hold phase: keep at 130°
  if (t >= v_t1 && t < v_t2) {
    ventAngle = v_contr;
  }

  // Relax phase: 130° → 90°
  float v_prog_relax = easedProgress(t, v_t2, timing.ventRelaxMs);
  if (t >= v_t2) {
    ventAngle = v_contr + (v_relaxed - v_contr) * v_prog_relax;
  }

  // Soft-start amplitude: shrink around base (keeps base at 90°, scales excursion)
  ventAngle = VENT_BASE_DEG + (ventAngle - VENT_BASE_DEG) * amp;

  // ---------- WRITE ----------
  writeServoAngle(SERVO_ATRIA_L, aL);
  writeServoAngle(SERVO_ATRIA_R, aR);
  writeServoAngle(SERVO_VENT,    ventAngle);

  // Overuse protection (kept): if something kept vent “contracted” too long, we’d trip emergency.
  // (No change needed here; your hard clamp to 70..130 already protects mechanics.)
}

// ============ Setup / loop ============
void setup() {
  pinMode(POT_PIN, INPUT);
  Serial.begin(115200);
  delay(50);
  Serial.println("Nilheim Heart – rev D′ (PCA9685 all, vent 90° base / 70–130° clamp)");

  Wire.begin();
  // 400 kHz if supported
  #if defined(ARDUINO_ARCH_AVR) || defined(ARDUINO_ARCH_SAMD) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    Wire.setClock(400000);
  #endif

  pwm.begin();
  pwm.setPWMFreq(SERVO_PWM_HZ);
  delay(10);

  // Power settle
  delay(POWER_ON_DELAY_MS);

  // Init pot filter & safety baseline
  potRaw = analogRead(POT_PIN);
  potEMA = (float)potRaw;
  lastPotRaw = potRaw;

  // Init timings & soft-start window
  computeTimings(bpmCurrent);
  cycleStartMs = millis();
  softStartBeginMs = cycleStartMs;
  softStartEndMs   = softStartBeginMs + SOFTSTART_RAMP_MS;

  // Park servos at relaxed positions (atria to their relaxed, vent to 90° base)
  writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
  writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
  writeServoAngle(SERVO_VENT,    VENT_BASE_DEG);
}

void loop() {
  uint32_t now = millis();
  if (now < lastLoopMs) lastLoopMs = 0; // handle wrap

  updateBPM(now);
  updateHeart(now);

  delay(2); // cooperative
}
