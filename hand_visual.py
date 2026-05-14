"""
Hand Tracking Generative Visual
================================
Move your fingers in front of your webcam — no touching needed.

Requirements:
    pip install opencv-python mediapipe numpy

Run:
    python hand_visual.py

Controls:
    Q or ESC  — quit
    1-5       — switch visual modes
    S         — save screenshot
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
WEBCAM_INDEX  = 0
SHOW_SKELETON = True   # overlay hand skeleton on visual
TRAIL_DECAY   = 0.88   # 0=no trail, 1=infinite trail
MODE_COUNT    = 5

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.55,
)

# ── State ─────────────────────────────────────────────────────────────────────
canvas      = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
mode        = 1
frame_count = 0
particles   = []   # list of dicts for mode 4 particle system
last_lm     = {}   # smoothed landmarks

# ── Helpers ───────────────────────────────────────────────────────────────────

def lm_px(lm, idx):
    """Return pixel coords (x, y) of landmark index."""
    p = lm.landmark[idx]
    return int(p.x * WIDTH), int(1 - p.y) * HEIGHT  # flip Y

def lm_norm(lm, idx):
    """Return normalized (x, y, z) of landmark index, Y flipped."""
    p = lm.landmark[idx]
    return p.x, 1.0 - p.y, p.z

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def lerp(a, b, t):
    return a + (b - a) * t

def smooth_lm(hand_id, lm):
    """Exponential smoothing on landmark positions."""
    key = hand_id
    raw = [(p.x, 1.0 - p.y, p.z) for p in lm.landmark]
    if key not in last_lm:
        last_lm[key] = raw
    else:
        last_lm[key] = [
            (lerp(last_lm[key][i][0], raw[i][0], 0.35),
             lerp(last_lm[key][i][1], raw[i][1], 0.35),
             lerp(last_lm[key][i][2], raw[i][2], 0.35))
            for i in range(21)
        ]
    return last_lm[key]

def norm_to_px(nx, ny):
    return int(nx * WIDTH), int(ny * HEIGHT)

def pinch_dist(lm_smooth):
    """Distance between index tip (8) and thumb tip (4), normalised 0-1."""
    ix, iy, _ = lm_smooth[8]
    tx, ty, _ = lm_smooth[4]
    return math.hypot(ix - tx, iy - ty)

def finger_spread(lm_smooth):
    """Average distance of finger tips from palm centre, 0-1."""
    tips = [4, 8, 12, 16, 20]
    cx, cy, _ = lm_smooth[0]
    dists = [math.hypot(lm_smooth[t][0] - cx, lm_smooth[t][1] - cy) for t in tips]
    return min(sum(dists) / len(dists) / 0.4, 1.0)

def hsv_color(h, s=1.0, v=1.0):
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p, q, t_ = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    seg = i % 6
    if   seg == 0: r,g,b = v,t_,p
    elif seg == 1: r,g,b = q,v,p
    elif seg == 2: r,g,b = p,v,t_
    elif seg == 3: r,g,b = p,q,v
    elif seg == 4: r,g,b = t_,p,v
    else:          r,g,b = v,p,q
    return (b, g, r)   # BGR for OpenCV

# ── Visual Modes ──────────────────────────────────────────────────────────────

def mode_ripple(canvas, all_lm):
    """Expanding ripple rings centred on index fingertips."""
    canvas *= TRAIL_DECAY
    t = frame_count * 0.04
    for lm_smooth in all_lm:
        ix, iy, iz = lm_smooth[8]
        px, py = norm_to_px(ix, iy)
        pd = pinch_dist(lm_smooth)
        spread = finger_spread(lm_smooth)
        hue = (ix + t * 0.1) % 1.0
        for ring in range(6):
            phase = (t + ring * 0.25) % 1.0
            radius = int(phase * 350 * (0.5 + spread))
            alpha  = (1.0 - phase) * (0.6 + 0.4 * (1 - pd))
            color  = hsv_color(hue + ring * 0.07)
            col_f  = tuple(c * alpha for c in color)
            tmp = np.zeros_like(canvas)
            cv2.circle(tmp, (px, py), radius, col_f, 2, cv2.LINE_AA)
            canvas += tmp

def mode_flow_field(canvas, all_lm):
    """Noise-like flowing lines steered by hand position."""
    canvas *= TRAIL_DECAY
    t = frame_count * 0.015
    if not all_lm:
        return
    lm_smooth = all_lm[0]
    cx, cy, _ = lm_smooth[9]   # palm centre
    spread = finger_spread(lm_smooth)
    pd = pinch_dist(lm_smooth)
    scale = 0.004 + spread * 0.008
    n_lines = 180
    for i in range(n_lines):
        angle_seed = i / n_lines * math.pi * 2
        px = cx + math.cos(angle_seed) * 0.45
        py = cy + math.sin(angle_seed) * 0.45
        hue = (i / n_lines + cx + t * 0.05) % 1.0
        color = hsv_color(hue, 0.8, 0.9)
        pts = []
        for _ in range(30):
            bx = int(px * WIDTH)
            by = int(py * HEIGHT)
            if not (0 <= bx < WIDTH and 0 <= by < HEIGHT):
                break
            pts.append((bx, by))
            noise_angle = (
                math.sin(px * 6 + t) * math.cos(py * 6 + t) +
                math.sin((px - cx) * 12) * 0.5
            ) * math.pi * 2
            speed = scale * (0.5 + pd * 1.5)
            px += math.cos(noise_angle) * speed
            py += math.sin(noise_angle) * speed
        if len(pts) > 1:
            overlay = np.zeros_like(canvas)
            for k in range(len(pts) - 1):
                cv2.line(overlay, pts[k], pts[k+1], color, 1, cv2.LINE_AA)
            canvas += overlay * 0.08

def mode_geometry(canvas, all_lm):
    """Rotating sacred geometry driven by finger angle and pinch."""
    canvas *= TRAIL_DECAY
    t = frame_count * 0.02
    for lm_smooth in all_lm:
        ix, iy, _ = lm_smooth[8]
        tx, ty, _ = lm_smooth[4]
        cx, cy, _ = lm_smooth[9]
        px, py = norm_to_px(cx, cy)
        pd = pinch_dist(lm_smooth)
        spread = finger_spread(lm_smooth)
        angle_offset = math.atan2(iy - cy, ix - cx)
        hue = (cx + t * 0.05) % 1.0
        for layer in range(4):
            sides = 3 + layer * 2
            radius = int((60 + layer * 55) * (0.4 + spread * 0.6))
            spin   = t * (0.5 + layer * 0.3) * (1 + pd) + angle_offset
            pts    = []
            for s in range(sides + 1):
                a = spin + s / sides * math.pi * 2
                pts.append((
                    px + int(math.cos(a) * radius),
                    py + int(math.sin(a) * radius)
                ))
            col = hsv_color(hue + layer * 0.12, 0.9, 0.95)
            overlay = np.zeros_like(canvas)
            for k in range(len(pts) - 1):
                cv2.line(overlay, pts[k], pts[k+1], col, 1, cv2.LINE_AA)
            canvas += overlay * 0.6

def mode_particles(canvas, all_lm):
    """Particle system emitted from fingertips."""
    global particles
    canvas *= TRAIL_DECAY
    t = frame_count * 0.04
    for lm_smooth in all_lm:
        tips = [4, 8, 12, 16, 20]
        pd = pinch_dist(lm_smooth)
        spread = finger_spread(lm_smooth)
        for tip_idx in tips:
            tx, ty, _ = lm_smooth[tip_idx]
            hue = (tip_idx / 20 + tx + t * 0.05) % 1.0
            for _ in range(int(3 + spread * 8)):
                angle = np.random.uniform(0, math.pi * 2)
                speed = np.random.uniform(0.002, 0.012) * (0.5 + spread)
                life  = np.random.randint(20, 60)
                particles.append({
                    'x': tx + np.random.uniform(-0.01, 0.01),
                    'y': ty + np.random.uniform(-0.01, 0.01),
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed - 0.004,
                    'life': life,
                    'max_life': life,
                    'hue': hue,
                    'size': np.random.randint(2, 5 + int(spread * 6)),
                })
    alive = []
    overlay = np.zeros_like(canvas)
    for p in particles:
        p['x']  += p['vx']
        p['y']  += p['vy']
        p['vy'] -= 0.0003
        p['life'] -= 1
        if p['life'] > 0 and 0 < p['x'] < 1 and 0 < p['y'] < 1:
            alpha = p['life'] / p['max_life']
            col   = hsv_color(p['hue'], 0.9, alpha)
            px, py = norm_to_px(p['x'], p['y'])
            cv2.circle(overlay, (px, py), p['size'], col, -1, cv2.LINE_AA)
            alive.append(p)
    particles = alive[-3000:]
    canvas += overlay * 0.9

def mode_warp_web(canvas, all_lm):
    """Web of lines between all hand landmarks, color by depth."""
    canvas *= TRAIL_DECAY
    t = frame_count * 0.025
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    overlay = np.zeros_like(canvas)
    for lm_smooth in all_lm:
        pd = pinch_dist(lm_smooth)
        for (a, b) in connections:
            ax, ay, az = lm_smooth[a]
            bx, by, bz = lm_smooth[b]
            pa = norm_to_px(ax, ay)
            pb = norm_to_px(bx, by)
            hue = ((az + bz) * 2 + ax + t * 0.06) % 1.0
            thick = max(1, int(3 - pd * 4))
            col = hsv_color(hue, 0.85, 0.95)
            cv2.line(overlay, pa, pb, col, thick, cv2.LINE_AA)
        # extra cross-finger lines for spread web
        tips = [4, 8, 12, 16, 20]
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                a, b = tips[i], tips[j]
                pa = norm_to_px(lm_smooth[a][0], lm_smooth[a][1])
                pb = norm_to_px(lm_smooth[b][0], lm_smooth[b][1])
                hue = (i * 0.15 + t * 0.04) % 1.0
                col = hsv_color(hue, 0.7, 0.6)
                cv2.line(overlay, pa, pb, col, 1, cv2.LINE_AA)
    canvas += overlay * 0.7

MODES = {
    1: ("Ripple Rings",   mode_ripple),
    2: ("Flow Field",     mode_flow_field),
    3: ("Sacred Geometry",mode_geometry),
    4: ("Particle Burst", mode_particles),
    5: ("Warp Web",       mode_warp_web),
}

# ── Skeleton overlay ──────────────────────────────────────────────────────────

def draw_skeleton(display, lm_list, hand_result):
    """Draw faint hand skeleton on the display frame."""
    for hand_landmarks in hand_result.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            display,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=2),
            mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1),
        )

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global canvas, mode, frame_count, particles, last_lm

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    cv2.namedWindow("Hand Visual", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Visual", WIDTH, HEIGHT)

    print("Hand Tracking Visual — running")
    print("  Keys: 1-5 = mode  |  S = screenshot  |  Q/ESC = quit")
    print(f"  Current mode: {MODES[mode][0]}")

    fps_time = time.time()
    fps_val  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands_model.process(rgb_frame)

        all_lm = []
        if result.multi_hand_landmarks:
            for hi, hand_lm in enumerate(result.multi_hand_landmarks):
                smoothed = smooth_lm(hi, hand_lm)
                all_lm.append(smoothed)
        else:
            last_lm.clear()

        # Run visual mode
        MODES[mode][1](canvas, all_lm)

        # Convert canvas to uint8 display
        display = np.clip(canvas * 255, 0, 255).astype(np.uint8)

        # Skeleton overlay
        if SHOW_SKELETON and result.multi_hand_landmarks:
            draw_skeleton(display, all_lm, result)

        # FPS counter
        now = time.time()
        fps_val = lerp(fps_val, 1.0 / max(now - fps_time, 1e-6), 0.1)
        fps_time = now
        frame_count += 1

        # HUD
        hud_col = (160, 160, 160)
        cv2.putText(display, f"Mode {mode}: {MODES[mode][0]}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_col, 1, cv2.LINE_AA)
        cv2.putText(display, f"{fps_val:.0f} fps",
                    (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_col, 1, cv2.LINE_AA)
        cv2.putText(display, "1-5: mode  S: screenshot  Q: quit",
                    (16, HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1, cv2.LINE_AA)

        cv2.imshow("Hand Visual", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            mode    = int(chr(key))
            canvas  = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
            particles.clear()
            last_lm.clear()
            print(f"  → Mode {mode}: {MODES[mode][0]}")
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"  → Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    print("Done.")

if __name__ == "__main__":
    main()
