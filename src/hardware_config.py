# hardware_config.py

# ---------------- INPUT PINS ----------------
BUTTON_PIN = 5
LIMIT_SWITCH_PIN = 6    # Servo limit switch (active LOW)

# ---------------- STEPPER X ----------------
STEPPER_X_DIR = 20
STEPPER_X_STEP = 21
STEPPER_X_EN   = 16

# ---------------- STEPPER Y ----------------
STEPPER_Y_DIR = 19
STEPPER_Y_STEP = 26
STEPPER_Y_EN   = 13

# motion parameters
STEPS_PER_MM_X = 80
STEPS_PER_MM_Y = 80

STEP_DELAY_SEC = 0.001

# ---------------- SERVO ----------------
SERVO_PIN = 17
SERVO_FREQ_HZ = 50
SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
