"Cosmotron — Robot controllers for Cosmotron 2025
Robot controllers for the Cosmotron 2025 simulation. This repo contains two Webots Python controllers:

open_cv.py — Rover controller: AprilTag search & approach, IMU-aware heading control, wheel steering, and emitter signalling.
arm_control — Arm (YouBot) controller: listens for START_ARM on Receiver channel 1 and executes a scripted pick/place sequence.
These controllers are intended to be run inside Webots as robot controllers (they import the Webots controller module)."

---

# Cosmotron — Robot controllers for Cosmotron 2025

This repository contains two Webots Python controllers used in the Cosmotron 2025 simulation:

- `open_cv.py` — Rover controller: AprilTag search & approach, IMU-aware heading control, wheel steering, and emitter signalling.
- `arm_control` — Arm (YouBot) controller: listens for `START_ARM` on Receiver channel 1 and executes a scripted pick/place sequence.

These controllers must be run by Webots (they import the Webots `controller` module).

