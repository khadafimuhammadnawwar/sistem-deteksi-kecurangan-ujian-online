---
name: 3d-gaze-proctoring
description: >
  Hardware-locked 3D Screen & Keyboard Gaze Ray-Casting and Anti-Spoof Lens Calibration
  for online exam proctoring using MediaPipe Tasks FaceLandmarker (478 landmarks + 4x4
  metric facial transformation matrix) and OpenCV. Distilled from PINTO0309/screen-eye-tracking,
  tensorsense/LaserGaze, and vardanagarwal/Proctoring-AI.
---

# 3D Gaze Ray-Casting & Anti-Spoof Proctoring

Use this skill whenever building, tuning, or auditing webcam eye-gaze and head-pose proctoring systems on laptops or desktop monitors.

## 1. Security Principle: Never Trust User-Calibrated Boundaries

- **Vulnerability (Calibration Poisoning / Spoofing):** Multi-point boundary calibration (asking a test-taker to look at Left, Right, Top, and Bottom edges) allows a dishonest user to look far outside the monitor during calibration, inflating the safe zone.
- **Defense (Hardware-Locked 3D Geometry + Anti-Spoof Lens Reference):**
  1. Lock physical device dimensions (`screen_width_cm`, `screen_height_cm`, `hinge_y_cm`, `touchpad_end_depth_cm`, `edge_margin_cm`, `lid_angle_deg`) in supervisor configuration (`DetectorConfig`).
  2. Use a **1-Point Physical Webcam Lens Check** `(0, 0, 0)` ONLY to measure personal anatomical Angle-Kappa offset (`±11.5°` yaw, `-12°..+24°` pitch max), filtering transient single blinks (`EAR >= min_calib_ear`) while requiring `>= min_valid_calib_samples` open-eye frames.
  3. Reject calibration immediately if head angle, iris socket ratio, eye aspect ratio (`EAR`), distance (`Z_cm`), or sample variance violates anatomical bounds.

## 2. Coordinate System & MediaPipe 3D Metric Extraction

- **Modern API Requirement (Python 3.12+ / MediaPipe 1.x):** Always use `mediapipe.tasks.python.vision.FaceLandmarker` (`RunningMode.VIDEO`, `output_facial_transformation_matrixes=True`). Never use removed `mp.solutions.face_mesh`.
- **Origin `(0, 0, 0)`:** Physical webcam lens at the top-center bezel of the display.
- **Metric Axes (on horizontally mirrored frame `cv2.flip(frame, 1)`):**
  - `X_cm = +M[0, 3]`: `-` User's Left, `+` User's Right.
  - `Y_cm = -M[1, 3]`: `-` Above webcam lens, `+` Downward toward screen/keyboard.
  - `Z_cm = abs(M[2, 3])`: Orthogonal distance from screen plane (`Z = 0`) toward user (`~35–60 cm`).
  - `head_yaw_deg = degrees(atan2(M[0, 2], M[2, 2]))`
  - `head_pitch_deg = degrees(atan2(-M[1, 2], M[2, 2]))` (`+` downward pitch, `-` upward pitch).

## 3. L-Shaped Laptop 3D Ray-Plane Intersection (`PINTO0309` + `LaserGaze`)

Combine 3D head pose angles with 478-landmark iris socket angular displacement (`468` & `473` relative to corners `33..133` and `362..263`), gated by `EAR >= min_ear_for_gaze` to suppress blink spikes while keeping head tracking active:

1. **Vertical Screen Plane (`Z = 0`):**
   - `screen_x_cm = eye_x_cm + eye_z_cm * tan(radians(total_yaw_deg))`
   - `screen_y_cm = eye_y_cm + eye_z_cm * tan(radians(total_pitch_deg))`
2. **Keyboard & Touchpad Deck Plane (`lid_angle_deg = 90° + alpha_tilt`, `Z in [0, touchpad_end_depth_cm]`):**
   - When `screen_y_cm > hinge_y_cm` and `sin(alpha_tilt) + cos(alpha_tilt) * tan_pitch > 0`:
     - `deck_z_cm = (screen_y_cm - hinge_y_cm) / (sin(alpha_tilt) + cos(alpha_tilt) * tan_pitch)` (`0 cm` continuously at screen hinge, `21.0 cm` at touchpad front edge)
     - `s_ray_cm = max(0.0, eye_z_cm - deck_z_cm * cos(alpha_tilt))`
     - `hit_x_cm = eye_x_cm + s_ray_cm * tan_yaw`
3. **Physical Trigger Rules:**
   - **Horizontal (`IND-01`):** `abs(hit_x_cm) > (screen_width_cm / 2.0 + edge_margin_cm)`
   - **Vertical Top (`IND-02`):** `screen_y_cm < -edge_margin_cm`
   - **Vertical Bottom / Touchpad End (`IND-02`):** `on_keyboard_deck and deck_z_cm >= touchpad_end_depth_cm`
