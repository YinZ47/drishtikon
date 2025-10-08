/*
  Nilheim Mechatronics Servo Heart – rev E (continuous-rotation ventricle safe drive)

  - Atria (ch0/ch1): positional, eased angles (0→1→0)
  - Ventricle (ch2): continuous-rotation speed control around neutral
      * forward during contraction → stop during hold → reverse during relax
      * balanced amplitudes so net rotation per beat ≈ 0 (prevents wind-up)
  - Safety: watchdog, pot spike detect, emergency stop with neutralization,
            per-update pulse cap, I2C deadband, soft-start, diastolic guard.

  Hardware:
  - PCA9685 @ I2C 0x40, 60 Hz
  - ch0 = Left Atrium (positional), ch1 = Right Atrium (positional), ch2 = Ventricle (MG90S 360°)
  - POT on A0 controls BPM
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>   // isfinite, fabsf

// ============ User configuration ============
constexpr uint8_t PCA9685_ADDR = 0x40;

// Servo channels
constexpr uint8_t SERVO_ATRIA_L = 0;
constexpr uint8_t SERVO_ATRIA_R = 1;
constexpr uint8_t SERVO_VENT    = 2;

constexpr uint8_t POT_PIN = A0;
constexpr float   SERVO_PWM_HZ = 60.0f;     // PCA9685 update rate

// Pot → BPM mapping & filtering
constexpr int     BPM_MIN = 45;
constexpr int     BPM_MAX = 140;
constexpr float   POT_EMA_ALPHA = 0.08f;
constexpr uint32_t POT_UPDATE_MS = 60;

// Command cadence
constexpr uint32_t SERVO_UPDATE_MS = 15; // ~66 Hz

// Soft-start
constexpr uint32_t POWER_ON_DELAY_MS = 1500; // allow rails to settle
constexpr uint32_t SOFTSTART_RAMP_MS = 1200; // amplitude ramp-in

// Physiology guards
constexpr uint32_t ATRIA_VENT_GAP_MS   = 30;
constexpr float    DIASTOLIC_MIN_FRAC  = 0.18f;
constexpr uint32_t DIASTOLIC_MIN_MS_LO = 120;
constexpr uint32_t DIASTOLIC_MIN_MS_HI = 400;

// I2C update suppression
constexpr uint16_t PWM_CHANGE_DEADBAND = 1;  // keep fine atrial steps

// ============ Safety / human-like motion ============
constexpr float    MAX_ANG_VEL_DEG_PER_SEC = 120.0f; // atria only
constexpr float    MIN_ANG_VEL_DEG_PER_SEC = 20.0f;  // atria nudge
constexpr uint16_t MAX_PULSE_CHANGE_PER_UPDATE = 40; // raw 0..4095 cap
constexpr uint32_t EMERGENCY_COOLDOWN_MS = 3000;
constexpr float    POT_SPIKE_THRESHOLD = 200.0f;
constexpr uint32_t MAX_LOOP_JITTER_MS  = 200;

// ============ Ventricle (continuous-rotation) tuning ============
// MG90S 360° typically stops near 1500 µs. Trim if needed.
constexpr bool     VENT_IS_CONTINUOUS   = true;
constexpr uint16_t VENT_NEUTRAL_US      = 1500; // adjust ±10–40 if it creeps
constexpr uint16_t VENT_SPEED_US_SPAN   = 180;  // max |offset| from neutral (safe)
constexpr uint16_t VENT_MIN_US          = 20;   // minimum |offset| to overcome deadband
constexpr float    VENT_SPEED_LIMIT     = 0.50f; // cap normalized |speed| (0..1)

// ============ Helpers ============
static inline float clampf(float v, float lo, float hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
static inline int   clampi(int v, int lo, int hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
static inline uint32_t clampu32(uint32_t v, uint32_t lo, uint32_t hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
static inline uint16_t absDiffU16(uint16_t a, uint16_t b){ return (a>b)?(uint16_t)(a-b):(uint16_t)(b-a); }
inline float easeSmoothStep(float x){ if(x<=0) return 0; if(x>=1) return 1; return x*x*(3.0f-2.0f*x); }

// convert microseconds → PCA9685 counts at SERVO_PWM_HZ
uint16_t usToCount(uint16_t us){
  float period_us = 1000000.0f / SERVO_PWM_HZ;   // e.g., 16666.7 µs at 60 Hz
  float duty = (float)us / period_us;            // 0..1
  int counts = (int)lroundf(duty * 4096.0f);     // 0..4096
  return (uint16_t)clampi(counts, 0, 4095);
}

// Per-servo calibration (atrial position range still used)
struct ServoCal {
  uint16_t minPulse;           // 0 deg
  uint16_t maxPulse;           // 180 deg
  float    relaxedAngleDeg;    // resting
  float    contractedAngleDeg; // peak contraction
};
ServoCal cal[3] = {
  { 140, 520,  25.0f,  55.0f }, // Left atrium
  { 140, 520,  25.0f,  55.0f }, // Right atrium
  { 140, 520,  35.0f, 115.0f }  // Ventricle (positional fields unused if continuous)
};

// ============ Internals ============
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

uint16_t lastPulse[3] = { 0xFFFF, 0xFFFF, 0xFFFF };
float    lastAngleDeg[3] = { -999.0f, -999.0f, -999.0f }; // atria only
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

// ============ Angle → pulse (atria) ============
uint16_t angleToPulse(uint8_t servoIdx, float angleDeg){
  angleDeg = clampf(angleDeg, 0.0f, 180.0f);
  const ServoCal &c = cal[servoIdx];
  float t = angleDeg / 180.0f;
  float pulse = c.minPulse + t * (float)(c.maxPulse - c.minPulse);
  int p = clampi((int)pulse, 0, 4095);
  return (uint16_t)p;
}

// Rate-limited, deadbanded, capped write (atria only)
void writeServoAngle(uint8_t servoIdx, float angleDeg){
  // Clamp near-calibrated range (small margin)
  const ServoCal &c = cal[servoIdx];
  float minA = c.relaxedAngleDeg - 10.0f;
  float maxA = c.contractedAngleDeg + 10.0f;
  angleDeg = clampf(angleDeg, minA, maxA);

  uint32_t now = millis();

  // Angular velocity & minimum step (atria)
  if (lastAngleDeg[servoIdx] > -900.0f){
    uint32_t dtMs = now - lastAngleTimeMs[servoIdx];
    if (dtMs > 0){
      float dtSec = dtMs / 1000.0f;
      float desiredDelta = angleDeg - lastAngleDeg[servoIdx];
      float ad = fabsf(desiredDelta);
      float maxDeltaDeg = MAX_ANG_VEL_DEG_PER_SEC * dtSec;
      float minDeltaDeg = MIN_ANG_VEL_DEG_PER_SEC * dtSec;

      if (ad > maxDeltaDeg){
        angleDeg = lastAngleDeg[servoIdx] + (desiredDelta > 0 ? maxDeltaDeg : -maxDeltaDeg);
      } else if (ad > 0.0f && ad < minDeltaDeg){
        float step = (ad < minDeltaDeg) ? ad : minDeltaDeg;
        angleDeg = lastAngleDeg[servoIdx] + (desiredDelta > 0 ? step : -step);
      }
    }
  }

  uint16_t pulse = angleToPulse(servoIdx, angleDeg);

  // Per-update pulse cap & deadband
  if (lastPulse[servoIdx] != 0xFFFF){
    int32_t diff = (int32_t)pulse - (int32_t)lastPulse[servoIdx];
    if (abs(diff) > (int)MAX_PULSE_CHANGE_PER_UPDATE){
      pulse = (uint16_t)((int32_t)lastPulse[servoIdx] + (diff > 0 ? MAX_PULSE_CHANGE_PER_UPDATE : - (int)MAX_PULSE_CHANGE_PER_UPDATE));
    }
    if (absDiffU16(lastPulse[servoIdx], pulse) < PWM_CHANGE_DEADBAND){
      return;
    }
  }

  pwm.setPWM(servoIdx, 0, pulse);
  lastPulse[servoIdx]      = pulse;
  lastAngleDeg[servoIdx]   = angleDeg;
  lastAngleTimeMs[servoIdx]= now;
}

// ============ CR ventricle: speed write around neutral ============
uint16_t ventNeutralCounts = 0;

void writeVentCRSpeed(float speedNorm){
  // speedNorm expected in [-1, 1]
  speedNorm = clampf(speedNorm, -VENT_SPEED_LIMIT, VENT_SPEED_LIMIT);

  // Convert normalized speed to µs offset
  float offset_us = speedNorm * (float)VENT_SPEED_US_SPAN;
  float mag = fabsf(offset_us);
  if (mag > 0 && mag < (float)VENT_MIN_US){
    offset_us = (offset_us > 0 ? (float)VENT_MIN_US : -(float)VENT_MIN_US);
  }

  int target = (int)ventNeutralCounts + (int)lroundf((offset_us / 1000000.0f) * SERVO_PWM_HZ * 4096.0f);
  target = clampi(target, 0, 4095);
  uint16_t pulse = (uint16_t)target;

  // Per-update cap and deadband (reuse generic guards)
  if (lastPulse[SERVO_VENT] != 0xFFFF){
    int32_t diff = (int32_t)pulse - (int32_t)lastPulse[SERVO_VENT];
    int32_t ad = (diff >= 0) ? diff : -diff;
    if (ad > (int32_t)MAX_PULSE_CHANGE_PER_UPDATE){
      pulse = (uint16_t)((int32_t)lastPulse[SERVO_VENT] + (diff > 0 ? (int32_t)MAX_PULSE_CHANGE_PER_UPDATE : -(int32_t)MAX_PULSE_CHANGE_PER_UPDATE));
    }
    if (absDiffU16(lastPulse[SERVO_VENT], pulse) < PWM_CHANGE_DEADBAND){
      return;
    }
  }

  pwm.setPWM(SERVO_VENT, 0, pulse);
  lastPulse[SERVO_VENT] = pulse;
}

// ============ Timing / physiology ============
void computeTimings(float bpm){
  bpm = clampf(bpm, (float)BPM_MIN, (float)BPM_MAX);
  uint32_t cycleMs = (uint32_t)(60000.0f / bpm);

  uint32_t diastolicMin = clampu32((uint32_t)(DIASTOLIC_MIN_FRAC * cycleMs),
                                   DIASTOLIC_MIN_MS_LO, DIASTOLIC_MIN_MS_HI);

  // AV delay and atria contraction
  uint32_t avDelay = (uint32_t)clampf(0.14f * cycleMs, 100.0f, 170.0f);
  uint32_t atriaContract = (uint32_t)clampf(0.16f * cycleMs, 100.0f, 180.0f);
  if (atriaContract + ATRIA_VENT_GAP_MS > avDelay){
    atriaContract = (avDelay > ATRIA_VENT_GAP_MS) ? (avDelay - ATRIA_VENT_GAP_MS) : (avDelay / 2);
    atriaContract = clampu32(atriaContract, 70, 180);
  }

  // Vent split
  uint32_t ventTotal   = (uint32_t)clampf(0.36f * cycleMs, 260.0f, 420.0f);
  uint32_t ventContract= (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);
  uint32_t ventHold    = (uint32_t)clampf(0.20f * ventTotal,  70.0f, 150.0f);
  uint32_t ventRelax   = (uint32_t)clampf(0.40f * ventTotal, 120.0f, 230.0f);

  uint32_t vEnd = avDelay + ventContract + ventHold + ventRelax;
  if (cycleMs - vEnd < diastolicMin){
    uint32_t need = diastolicMin - (cycleMs - vEnd);
    uint32_t shrinkable = ventContract + ventHold + ventRelax;
    if (need < shrinkable){
      float s = (float)(shrinkable - need) / (float)shrinkable;
      ventContract = clampu32((uint32_t)(ventContract * s), 100, 220);
      ventHold     = clampu32((uint32_t)(ventHold     * s),  60, 140);
      ventRelax    = clampu32((uint32_t)(ventRelax    * s), 100, 220);
    } else {
      ventContract = 100; ventHold = 60; ventRelax = 100;
    }
    vEnd = avDelay + ventContract + ventHold + ventRelax;
  }

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

float easedProgress(uint32_t nowInCycleMs, uint32_t t0, uint32_t durMs){
  if (nowInCycleMs <= t0) return 0.0f;
  uint32_t dt = nowInCycleMs - t0;
  if (dt >= durMs) return 1.0f;
  float x = (float)dt / (float)durMs;
  return easeSmoothStep(x);
}

float softstartScale(uint32_t nowMs){
  if (nowMs <= softStartBeginMs) return 0.0f;
  if (nowMs >= softStartEndMs)   return 1.0f;
  float x = (float)(nowMs - softStartBeginMs) / (float)(softStartEndMs - softStartBeginMs);
  return easeSmoothStep(x);
}

// bump shape for speed (0 at edges, 1 in middle)
inline float bumpShape01(float x){ if (x<=0.0f || x>=1.0f) return 0.0f; return 6.0f*x*(1.0f - x); }
// normalized phase position in [0,1]
inline float phase01(uint32_t t, uint32_t start, uint32_t dur){
  if (t <= start) return 0.0f;
  if (t >= start + dur) return 1.0f;
  return (float)(t - start) / (float)dur;
}

// ============ Safety helpers ============
void enterEmergencyStop(const char *reason){
  if (!emergencyStopActive){
    emergencyStopActive = true;
    emergencyStopSince = millis();
    Serial.print("EMERGENCY STOP: ");
    Serial.println(reason);
    // Atria to relaxed; vent to neutral (stop)
    writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
    writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
    writeVentCRSpeed(0.0f);
  }
}

void clearEmergencyIfSafe(){
  if (!emergencyStopActive) return;
  if (millis() - emergencyStopSince < EMERGENCY_COOLDOWN_MS) return;
  if (abs((int)potRaw - (int)lastPotRaw) < 10){
    emergencyStopActive = false;
    Serial.println("EMERGENCY CLEARED: pot stable and cooldown elapsed.");
  } else {
    lastPotRaw = potRaw;
  }
}

// ============ Input / BPM update ============
void updateBPM(uint32_t nowMs){
  if (nowMs - lastPotSampleMs >= POT_UPDATE_MS){
    lastPotSampleMs = nowMs;

    uint16_t raw = analogRead(POT_PIN);
    potRaw = raw;
    if (potEMA <= 0.01f) potEMA = (float)raw;
    potEMA = potEMA + POT_EMA_ALPHA * ((float)raw - potEMA);

    if (abs((int)raw - (int)lastPotRaw) > (int)POT_SPIKE_THRESHOLD){
      enterEmergencyStop("pot spike detected");
    }
    lastPotRaw = raw;

    float targetBPM = BPM_MIN + (BPM_MAX - BPM_MIN) * (potEMA / 1023.0f);

    static uint32_t lastMs = nowMs;
    float dtSec = (nowMs - lastMs) / 1000.0f;
    if (dtSec < 0.0001f) dtSec = 0.0001f;
    lastMs = nowMs;

    float maxDelta = BPM_SLEW_PER_SEC * dtSec;
    float delta = targetBPM - bpmCurrent;
    if (delta >  maxDelta) delta =  maxDelta;
    if (delta < -maxDelta) delta = -maxDelta;
    bpmCurrent += delta;

    if (!isfinite(potEMA) || potEMA < -100.0f || potEMA > 5000.0f){
      enterEmergencyStop("pot EMA invalid");
    }
  }
}

// ============ Heart update ============
void updateHeart(uint32_t nowMs){
  // watchdog
  if (lastLoopMs != 0 && nowMs - lastLoopMs > MAX_LOOP_JITTER_MS){
    enterEmergencyStop("loop jitter too large");
  }
  lastLoopMs = nowMs;

  if (emergencyStopActive){
    clearEmergencyIfSafe();
    return;
  }

  // cycle boundary
  if (nowMs - cycleStartMs >= timing.cycleMs){
    cycleStartMs = nowMs;
    computeTimings(bpmCurrent);
  }

  // command throttle
  if (nowMs - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  uint32_t dtServoUpdate = nowMs - lastServoUpdateMs;
  lastServoUpdateMs = nowMs;

  uint32_t t = nowMs - cycleStartMs;
  float amp = softstartScale(nowMs);

  // ---------- ATRIA (shared activation → per-servo mapping) ----------
  float a_prog_contract = easedProgress(t, 0,                      timing.atriaContractMs);
  float a_prog_relax    = easedProgress(t, timing.atriaRelaxStartMs, timing.atriaRelaxMs);
  float a_act = a_prog_contract + (0.0f - a_prog_contract) * a_prog_relax; // 0→1→0
  a_act = clampf(a_act, 0.0f, 1.0f) * amp;

  float aL = cal[SERVO_ATRIA_L].relaxedAngleDeg +
             (cal[SERVO_ATRIA_L].contractedAngleDeg - cal[SERVO_ATRIA_L].relaxedAngleDeg) * a_act;
  float aR = cal[SERVO_ATRIA_R].relaxedAngleDeg +
             (cal[SERVO_ATRIA_R].contractedAngleDeg - cal[SERVO_ATRIA_R].relaxedAngleDeg) * a_act;

  writeServoAngle(SERVO_ATRIA_L, aL);
  writeServoAngle(SERVO_ATRIA_R, aR);

  // ---------- VENTRICLE (continuous rotation) ----------
  // phase times
  uint32_t v_t0 = timing.avDelayMs;
  uint32_t v_t1 = v_t0 + timing.ventContractMs;
  uint32_t v_t2 = v_t1 + timing.ventHoldMs;
  uint32_t v_t3 = v_t2 + timing.ventRelaxMs; (void)v_t3;

  // speed bump during contract (forward) and relax (reverse)
  float x_c = phase01(t, v_t0, timing.ventContractMs);
  float x_r = phase01(t, v_t2, timing.ventRelaxMs);
  float bump_c = bumpShape01(x_c);
  float bump_r = bumpShape01(x_r);

  // balance areas so net rotation per beat ≈ 0 even if durations differ
  float relaxGain = (timing.ventContractMs > 0 && timing.ventRelaxMs > 0)
                      ? ((float)timing.ventContractMs / (float)timing.ventRelaxMs)
                      : 1.0f;

  float ventSpeed = 0.0f;
  if (t >= v_t0 && t < v_t1) {
    ventSpeed = +amp * bump_c;                 // forward
  } else if (t >= v_t1 && t < v_t2) {
    ventSpeed = 0.0f;                          // hold (stop)
  } else if (t >= v_t2) {
    ventSpeed = -amp * bump_r * relaxGain;     // reverse
  }
  // scale to limit
  ventSpeed = clampf(ventSpeed, -VENT_SPEED_LIMIT, VENT_SPEED_LIMIT);

  writeVentCRSpeed(ventSpeed);

  // Optional anomaly guard: if commanded motion is almost always forward, stop
  static uint32_t ventActiveMs = 0;
  if (fabsf(ventSpeed) > 0.05f) ventActiveMs += dtServoUpdate; else ventActiveMs = 0;
  if (ventActiveMs > (uint32_t)(timing.cycleMs * 0.9f)) {
    enterEmergencyStop("prolonged vent motion detected");
    return;
  }
}

// ============ Setup / loop ============
void setup(){
  pinMode(POT_PIN, INPUT);
  Serial.begin(115200);
  delay(50);
  Serial.println("Nilheim Heart – rev E (atrium position + CR ventricle)");

  Wire.begin();
  #if defined(ARDUINO_ARCH_AVR) || defined(ARDUINO_ARCH_SAMD) || defined(ARDUINO_ARCH_ESP32) || defined(ARDUINO_ARCH_RP2040)
    Wire.setClock(400000);
  #endif

  pwm.begin();
  pwm.setPWMFreq(SERVO_PWM_HZ);
  delay(10);

  // compute neutral counts once
  ventNeutralCounts = usToCount(VENT_NEUTRAL_US);

  delay(POWER_ON_DELAY_MS); // power settle

  // init pot filter & safety baseline
  potRaw = analogRead(POT_PIN);
  potEMA = (float)potRaw;
  lastPotRaw = potRaw;

  // timings & soft-start
  computeTimings(bpmCurrent);
  cycleStartMs = millis();
  softStartBeginMs = cycleStartMs;
  softStartEndMs   = softStartBeginMs + SOFTSTART_RAMP_MS;

  // park atria at relaxed; vent at neutral (stop)
  writeServoAngle(SERVO_ATRIA_L, cal[SERVO_ATRIA_L].relaxedAngleDeg);
  writeServoAngle(SERVO_ATRIA_R, cal[SERVO_ATRIA_R].relaxedAngleDeg);
  writeVentCRSpeed(0.0f);
}

void loop(){
  uint32_t now = millis();
  if (now < lastLoopMs) lastLoopMs = 0; // wrap guard

  updateBPM(now);
  updateHeart(now);

  delay(2); // cooperative
}
