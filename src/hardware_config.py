# hardware_config.py

# ---------------- INPUT PINS ----------------
BUTTON_PIN = 5
LIMIT_SWITCH_PIN = 4    # Servo limit switch (active LOW)

# ---------------- Camera -------------------
IR_PIN = 26

# ---------------- STEPPER X ----------------
STEPPER_X_DIR = 12
STEPPER_X_STEP = 5

# ---------------- STEPPER Y ----------------
STEPPER_Y_DIR = 19
STEPPER_Y_STEP = 13

# motion parameters
MICROSTEP_X = 4
MICROSTEP_Y = 4
MOTOR_STEP = 360 / 1.8

STEP_DELAY_SEC = 0.001

HOMING_X_LIMIT_SWITCH_PIN = 23
HOMING_Y_LIMIT_SWITCH_PIN = 24

# ---------------- SERVO ----------------
SERVO_PIN = 18
SERVO_FREQ_HZ = 50
SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
