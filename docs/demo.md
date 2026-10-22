# Tactical HUD Interface & Telemetry Demo

This document outlines the visual layout, user feedback states, and telemetry instrumentation of the **Human Activity Recognition & Posture Classification HUD**.

---

## 1. HUD Visual Concept & Architecture

The application serves a zero-dependency HTML5 Canvas and CSS HUD interface directly from FastAPI (`GET /`). Designed with a futuristic cyber-tactical aesthetic, the interface provides immediate cognitive feedback for ergonomics and activity tracking.

```
+-----------------------------------------------------------------------------+
|  HAR NEURAL HUD  ::  POSTURE & ACTIVITY RECOGNITION V2.4       [STATUS: OK] |
+------------------------------------+----------------------------------------+
|                                    | TELEMETRY GAUGES                       |
|                                    | -------------------------------------- |
|                                    | POSTURE STATE:     [  SLOUCHING  ]     |
|          WEBCAM FEED               | CONFIDENCE:        [ 91.4% █████████ ] |
|               &                    | BUFFER CAPACITY:   [ 64/64 █████████ ] |
|      CANVAS SKELETON               | INFERENCE LATENCY: [ 0.28 ms         ] |
|          OVERLAY                   | -------------------------------------- |
|                                    | BIOMECHANICAL DIAGNOSTICS              |
|        (10 Keypoints)              | - Torso Incline:   18.4 deg            |
|                                    | - Knee Angle:      89.2 deg (Seated)   |
|                                    | - Spinal Curvature: HIGH               |
|                                    | -------------------------------------- |
|                                    | [!] ERGONOMIC WARNING:                 |
|                                    | >> FORWARD HEAD TILT DETECTED <<       |
|                                    | PLEASE STRAIGHTEN YOUR SPINE           |
+------------------------------------+----------------------------------------+
|  FRAME BUFFER: [================ 64/64]  |  SESSION ID: a4f8-9e12-bb20-4411 |
+-----------------------------------------------------------------------------+
```

---

## 2. Interactive Telemetry States

The HUD dynamically cycles through four distinct operational states based on camera feed and buffer accumulation:

### State 1: Human Search & Zero Detection
- **Trigger:** Camera feed is active, but MediaPipe Pose fails to detect a human subject in the frame.
- **HUD Response:**
  - Status Indicator: Amber flashing `SEARCHING FOR SUBJECT...`
  - Skeleton Overlay: Cleared
  - Session Buffer: Resets to `0/64` to prevent corrupted temporal sequences.

### State 2: Temporal Buffer Ramp-Up
- **Trigger:** A subject enters the frame. MediaPipe extracts valid 10-landmark coordinates.
- **HUD Response:**
  - Status Indicator: Cyan `DETECTING MOTION DYNAMICS... (k / 64)`
  - Progress Gauge: Increments smoothly from $1$ to $64$.
  - Dynamic Feedback: Visual keypoints render on canvas with connecting green limb vectors.

### State 3: Validated Posture Recognition & Ergonomic Alert
- **Trigger:** Buffer reaches 64 frames and the neural network predicts **Slouching** with confidence $> 80\%$.
- **HUD Response:**
  - Status Indicator: High-contrast red/amber `SLOUCHING DETECTED`
  - Ergonomic Alert: Audio chime and tactical visual warning banner urging spinal realignment.
  - Telemetry: Displays measured forward head displacement and thoracic spine curvature.

### State 4: Continuous Activity Locomotion
- **Trigger:** Subject engages in dynamic motion (**Walking** or **Running**).
- **HUD Response:**
  - Status Indicator: Bright Emerald Green `ACTIVE: WALKING` or `ACTIVE: RUNNING`
  - Gait Cadence: Displays estimated step cadence derived from alternating knee-hip vectors.
  - Energy Expenditure: Real-time METs (Metabolic Equivalent of Task) approximation.

---

## 3. Skeleton Canvas Rendering Protocol

To achieve 60 FPS rendering in the browser without GPU overhead:
- **Landmark Anchors:** 10 tracked keypoints rendered as glowing 4px circles with distinct chromatic palettes (green for upper torso, cyan for limbs).
- **Vector Bone Connections:** Line segments connecting $(x_1, y_1) \to (x_2, y_2)$ using hardware-accelerated 2D canvas primitives.
- **Occlusion Damping:** Joint confidence scores beneath $0.5$ render as dashed translucent lines, signaling visual obstruction to the user.

---

## 4. Architectural Separation & Security

The frontend HUD operates entirely statelessly with respect to the user's video feed:
- **Client-Side Vision:** MediaPipe processes video frames locally in the browser's WebAssembly sandbox.
- **No Video Streaming:** Raw webcam frames are never transmitted to the backend server. Only anonymized $10 \times 3$ coordinate arrays are sent over the API, ensuring total privacy.
