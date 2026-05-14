[README.md](https://github.com/user-attachments/files/27746510/README.md)
# Hand Tracking Generative Visual

Move your fingers in front of your webcam — no touching needed.
The visuals react in real time to your finger positions, pinch gestures, and hand spread.

## Quick Start

### 1. Install dependencies
```
pip install opencv-python mediapipe numpy
```

### 2. Run
```
python hand_visual.py
```

## Controls

| Key     | Action                         |
|---------|--------------------------------|
| `1`     | Ripple Rings                   |
| `2`     | Flow Field                     |
| `3`     | Sacred Geometry                |
| `4`     | Particle Burst                 |
| `5`     | Warp Web (full skeleton)       |
| `S`     | Save screenshot as PNG         |
| `Q/ESC` | Quit                           |

## What Each Mode Does

| Mode | Visual | Driven by |
|------|--------|-----------|
| 1 Ripple Rings    | Expanding colored rings | Index fingertip XY, pinch distance |
| 2 Flow Field      | Swirling particle lines | Palm centre, finger spread |
| 3 Sacred Geometry | Rotating polygons       | Finger angle, pinch distance |
| 4 Particle Burst  | Emitted particles       | All 5 fingertips + spread |
| 5 Warp Web        | Glowing hand skeleton   | All 21 landmarks + depth |

## Gesture Reference

- **Pinch** (index ↔ thumb close) — shrinks/changes effect density
- **Spread fingers wide** — expands effect size and intensity
- **Move hand left/right** — shifts hue/color
- **Move hand up/down** — changes speed or brightness
- **Two hands** — double the effect (modes 1, 4, 5 support two hands)

## Tweaking

Open `hand_visual.py` and edit the Config section at the top:

```python
TRAIL_DECAY   = 0.88   # 0 = no trail, 1 = infinite trail
SHOW_SKELETON = True   # overlay hand wireframe
WEBCAM_INDEX  = 0      # change if you have multiple cameras
```

## Lighting Tips

- Works best with decent lighting on your hand
- Plain background helps detection
- If tracking is unreliable, lower `min_detection_confidence` to `0.55` in the script
