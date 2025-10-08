/*
  Nilheim Mechatronics Servo Heart – biologically sequenced driver (rev D: merged + protected)

  This version merges:
  - Your "rev B (safer)" protections (angular-velocity limit, pulse change cap, watchdogs, emergency stop)
  - Biologically friendly sequencing (atria → AV gap → ventricle) with diastolic-reserve guards
  - Soft-start (power settle + amplitude ramp)
  - Per-atria activation mapping (each atrium uses its own calibration)  <-- important fix
  - Finer I2C deadband so small atrial steps go through

  Hardware:
  - PCA9685 (I2C, default 0x40)
  - Servos: ch0 = Left Atrium, ch1 = Right Atrium, ch2 = Ventricle
  - Pot on A0 for BPM
*/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>   // for isfinite, fabsf

// ============ User configuration ============
constexpr uint8_t PCA9685_ADDR = 0x40;

// Servo channels (as requested)
constexpr uint8_t SERVO_ATRIA_L = 0;
constexpr uint8_t SERVO_ATRIA_R = 1;
constexpr uint8_t SERVO_VENT    = 2;

constexpr uint8_t POT_PIN = A0;
constexpr float   SERVO_PWM_HZ = 60.0f;

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
constexpr uint16_t PWM_CHANGE_DEADBAND = 1;      // finer than 2 so atrial micro-steps pass

// ============ Safety / human-like motion ============
constexpr float    MAX_ANG_VEL_DEG_PER_SEC = 120.0f; // clamp overly fast moves
constexpr float    MIN_ANG_VEL_DEG_PER_SEC = 20.0f;  // ensure tiny moves don't stall
constexpr uint16_t MAX_PULSE_CHANGE_PER_UPDATE = 40; // cap raw 0..4095 change per write
constexpr uint32_t EMERGENCY_COOLDOWN_MS = 3000;
constexpr float    POT_SPIKE_THRESHOLD = 200.0f;     // sudden raw jump triggers stop
constexpr uint32_t MAX_LOOP_JITTER_MS  = 200;        // watchdog

// ============ Helpers ============
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static inline int clampi(int v, int lo
