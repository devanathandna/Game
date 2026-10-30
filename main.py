"""
HandRush - a hand-gesture arcade game (OpenCV + MediaPipe)

ONE window: the game is the main stage, your webcam is a Zoom-style
self-view tile in the bottom-right corner (mirrored, rounded, with a LIVE badge).

GESTURES
  Point (index finger) ... move the catcher
  Pinch (thumb + index) .. slow-motion (drains the energy bar)
  Open palm (hold 1s) .... start / restart / resume
  Fist (hold 0.7s) ....... pause

KEYS (fallbacks)
  SPACE start/pause   R restart   C hide/show camera tile
  F fullscreen        Q / ESC quit

ITEMS
  Blue orb +10 | Gold orb +30 | Bomb = lose a life (shield absorbs it)
  Pink + = extra life | Blue S = shield | Purple T = slow-mo | Orange 2x = double points

Install:  pip install opencv-python mediapipe==0.10.14 numpy
"""

import json
import math
import os
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------- config
W, H = 1280, 720                       # game stage size
TILE_W, TILE_H, MARGIN = 320, 240, 24  # Zoom-style camera tile
CATCHER_R = 46
MAX_LIVES = 5
SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "handrush_save.json")
FONT = cv2.FONT_HERSHEY_DUPLEX

# BGR colours
KINDS = {
    "orb":    dict(w=58, r=26, col=(255, 200, 60),  label=""),
    "gold":   dict(w=10, r=22, col=(0, 215, 255),   label="$"),
    "bomb":   dict(w=18, r=28, col=(45, 45, 55),    label=""),
    "heart":  dict(w=3,  r=24, col=(180, 105, 255), label="+"),
    "shield": dict(w=4,  r=24, col=(255, 150, 50),  label="S"),
    "slow":   dict(w=4,  r=24, col=(230, 90, 190),  label="T"),
    "double": dict(w=4,  r=24, col=(0, 140, 255),   label="2x"),
}


# ----------------------------------------------------------------- helpers
def ip(x, y):
    return int(x), int(y)


def put_text(img, s, pos, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = pos
    if center:
        x -= tw // 2
        y += th // 2
    x, y = int(x), int(y)
    cv2.putText(img, s, (x + 2, y + 2), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for c in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, c, r, 255, -1)
    return m > 0


def dim(img, amount=0.55):
    return cv2.addWeighted(img, 1 - amount, np.zeros_like(img), amount, 0)


def load_best():
    try:
        with open(SAVE_FILE) as f:
            return int(json.load(f).get("best", 0))
    except Exception:
        return 0


def save_best(v):
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump({"best": int(v)}, f)
    except Exception:
        pass


# ----------------------------------------------------------------- hand tracking
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.pos = np.array([W / 2, H / 2], dtype=float)
        self.present = False
        self.last_seen = time.perf_counter()
        self.gesture = "none"
        self.pinch_ratio = 1.0

    @staticmethod
    def _d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _classify(self, p):
        d = self._d
        wrist = p[0]
        ext = sum(d(p[t], wrist) > d(p[pip], wrist) * 1.05
                  for t, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
        scale = d(p[0], p[9]) or 1e-6
        self.pinch_ratio = d(p[4], p[8]) / scale
        if self.pinch_ratio < 0.28:
            return "pinch"
        if ext >= 4:
            return "palm"
        if ext == 0:
            return "fist"
        return "point"

    def update(self, cam_bgr):
        """Processes a (already mirrored) frame and draws the skeleton on it."""
        rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            p = hand.landmark
            self.present = True
            self.last_seen = time.perf_counter()
            # map the central part of the camera view to the whole stage
            tx = float(np.clip((p[8].x - 0.12) / 0.76, 0, 1)) * W
            ty = float(np.clip((p[8].y - 0.10) / 0.70, 0, 1)) * H
            target = np.array([tx, ty])
            dist = float(np.linalg.norm(target - self.pos))
            alpha = float(np.clip(dist / 160.0, 0.25, 0.85))  # adaptive smoothing
            self.pos += (target - self.pos) * alpha
            self.gesture = self._classify(p)
            self.drawer.draw_landmarks(
                cam_bgr, hand, self.mp_hands.HAND_CONNECTIONS,
                self.drawer.DrawingSpec(color=(0, 255, 180), thickness=3, circle_radius=3),
                self.drawer.DrawingSpec(color=(255, 255, 255), thickness=2),
            )
        else:
            self.present = False
            self.gesture = "none"


# ----------------------------------------------------------------- game logic
class Game:
    def __init__(self):
        self.best = load_best()
        rng = np.random.RandomState(7)
        self.stars = [[rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 70), rng.randint(1, 3)]
                      for _ in range(90)]
        top = np.array([48, 20, 20], np.float32)
        bot = np.array([22, 6, 40], np.float32)
        grad = np.linspace(top, bot, H)[:, None, :].repeat(W, axis=1)
        self.bg = grad.astype(np.uint8)
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = 3
        self.level = 1
        self.combo = 0
        self.best_combo = 0
        self.caught = 0
        self.misses = 0
        self.energy = 0.0
        self.shield = False
        self.slow_t = 0.0
        self.double_t = 0.0
        self.items, self.particles, self.popups = [], [], []
        self.trail = deque(maxlen=14)
        self.spawn_t = 1.0
        self.t = 0.0
        self.flash = 0.0
        self.shake = 0.0
        self.pinch_active = False
        self.over = False
        self.new_record = False

    # ---- helpers
    def multiplier(self):
        return min(5, 1 + self.combo // 5)

    def burst(self, x, y, col, n=18):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(80, 340)
            life = random.uniform(0.4, 0.9)
            self.particles.append([x, y, math.cos(a) * sp, math.sin(a) * sp, life, life, col,
                                   random.randint(2, 5)])

    def popup(self, x, y, s, col=(255, 255, 255), size=0.9):
        self.popups.append([x, y, s, 1.0, col, size])

    def lose_life(self):
        self.lives -= 1
        self.flash, self.shake, self.combo = 0.5, 0.4, 0
        if self.lives <= 0:
            self.over = True
            if self.score > self.best:
                self.best, self.new_record = self.score, True
                save_best(self.best)

    def spawn(self):
        kinds = list(KINDS)
        weights = [KINDS[k]["w"] + (self.level * 1.5 if k == "bomb" else 0) for k in kinds]
        k = random.choices(kinds, weights)[0]
        r = KINDS[k]["r"]
        speed = 210 + self.level * 38 + random.uniform(-30, 60)
        self.items.append(dict(k=k, x=random.uniform(r + 20, W - r - 20), y=-r,
                               vy=speed, r=r, rot=random.uniform(0, 6)))

    # ---- update
    def update(self, dt, cur, pinching):
        self.t += dt
        scale = 1.0
        if self.slow_t > 0:
            self.slow_t -= dt
            scale = 0.5
        self.pinch_active = bool(pinching and self.energy > 0)
        if self.pinch_active:
            self.energy = max(0.0, self.energy - 28 * dt)
            scale = min(scale, 0.4)
        self.double_t = max(0.0, self.double_t - dt)
        self.flash = max(0.0, self.flash - dt)
        self.shake = max(0.0, self.shake - dt)
        sdt = dt * scale

        self.trail.append(ip(*cur))

        self.spawn_t -= sdt
        if self.spawn_t <= 0:
            self.spawn()
            self.spawn_t = max(0.22, 0.85 - 0.06 * (self.level - 1)) * random.uniform(0.7, 1.3)

        alive = []
        for it in self.items:
            it["y"] += it["vy"] * sdt
            it["rot"] += 2.5 * sdt
            if math.hypot(it["x"] - cur[0], it["y"] - cur[1]) < it["r"] + CATCHER_R:
                self.on_catch(it)
                continue
            if it["y"] > H + 40:
                if it["k"] in ("orb", "gold"):
                    self.misses += 1
                    self.combo = 0
                    if self.misses >= 5:
                        self.misses = 0
                        self.popup(it["x"], H - 60, "5 MISSES!", (80, 80, 255))
                        self.lose_life()
                continue
            alive.append(it)
        self.items = alive

        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 400 * dt
            p[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]
        for pu in self.popups:
            pu[1] -= 60 * dt
            pu[3] -= dt * 0.9
        self.popups = [pu for pu in self.popups if pu[3] > 0]
        for s in self.stars:
            s[1] += s[2] * dt * (0.5 if scale < 1 else 1.0)
            if s[1] > H:
                s[0], s[1] = random.uniform(0, W), 0

    def on_catch(self, it):
        k, x, y = it["k"], it["x"], it["y"]
        col = KINDS[k]["col"]
        if k == "bomb":
            if self.shield:
                self.shield = False
                self.burst(x, y, (255, 150, 50), 30)
                self.popup(x, y, "BLOCKED", (255, 180, 80))
            else:
                self.burst(x, y, (40, 40, 255), 36)
                self.popup(x, y, "BOOM!", (60, 60, 255), 1.2)
                self.lose_life()
            return
        self.burst(x, y, col)
        if k in ("orb", "gold"):
            base = 10 if k == "orb" else 30
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            self.caught += 1
            gained = base * self.multiplier() * (2 if self.double_t > 0 else 1)
            self.score += gained
            self.energy = min(100.0, self.energy + (4 if k == "orb" else 10))
            self.popup(x, y, f"+{gained}", col)
            new_level = 1 + self.score // 200
            if new_level > self.level:
                self.level = new_level
                self.popup(W / 2, H / 2 - 80, f"LEVEL {self.level}", (0, 255, 200), 1.8)
        elif k == "heart":
            self.lives = min(MAX_LIVES, self.lives + 1)
            self.popup(x, y, "+1 LIFE", col)
        elif k == "shield":
            self.shield = True
            self.popup(x, y, "SHIELD", col)
        elif k == "slow":
            self.slow_t = 5.0
            self.popup(x, y, "SLOW-MO", col)
        elif k == "double":
            self.double_t = 8.0
            self.popup(x, y, "2x POINTS", col)

    # ---- drawing
    def draw_item(self, img, it):
        k = KINDS[it["k"]]
        x, y = ip(it["x"], it["y"])
        r = it["r"]
        if it["k"] == "bomb":
            cv2.circle(img, (x, y), r, k["col"], -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, (40, 40, 255), 3, cv2.LINE_AA)
            for a in (it["rot"], it["rot"] + math.pi / 2):
                dx, dy = math.cos(a) * r * 0.55, math.sin(a) * r * 0.55
                cv2.line(img, ip(x - dx, y - dy), ip(x + dx, y + dy), (60, 60, 255), 3, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), r, k["col"], -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (x - r // 3, y - r // 3), max(2, r // 5), (255, 255, 255), -1, cv2.LINE_AA)
            if k["label"]:
                put_text(img, k["label"], (x + 2, y + 2), 0.65, (255, 255, 255), 2, center=True)

    def draw_world(self, tracker):
        f = self.bg.copy()
        for s in self.stars:
            c = 120 + s[3] * 40
            cv2.circle(f, ip(s[0], s[1]), s[3], (c, c, c), -1)

        # glow layer (one blend per frame)
        ov = f.copy()
        for it in self.items:
            gc = (40, 40, 255) if it["k"] == "bomb" else KINDS[it["k"]]["col"]
            cv2.circle(ov, ip(it["x"], it["y"]), int(it["r"] * 1.8), gc, -1, cv2.LINE_AA)
        f = cv2.addWeighted(ov, 0.28, f, 0.72, 0)
        for it in self.items:
            self.draw_item(f, it)

        # trail
        pts = list(self.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(f, pts[i - 1], pts[i], (int(255 * a), int(220 * a), int(90 * a)),
                     max(1, int(14 * a)), cv2.LINE_AA)

        for p in self.particles:
            a = p[4] / p[5]
            col = tuple(int(c * a) for c in p[6])
            cv2.circle(f, ip(p[0], p[1]), p[7], col, -1)

        # catcher / cursor
        cx, cy = ip(*tracker.pos)
        ring = (230, 90, 190) if self.pinch_active else (255, 255, 120)
        if not tracker.present:
            ring = (120, 120, 120)
        cv2.circle(f, (cx, cy), CATCHER_R, ring, 3, cv2.LINE_AA)
        cv2.circle(f, (cx, cy), 6, ring, -1, cv2.LINE_AA)
        if self.shield:
            pulse = int(8 * math.sin(self.t * 6))
            cv2.circle(f, (cx, cy), CATCHER_R + 14 + pulse // 2, (255, 150, 50), 3, cv2.LINE_AA)

        for x, y, s, life, col, size in self.popups:
            c = tuple(int(v * min(1.0, life * 1.5)) for v in col)
            put_text(f, s, (x, y), size, c, 2, center=True)

        if self.shake > 0:
            m = self.shake * 30
            M = np.float32([[1, 0, random.uniform(-m, m)], [0, 1, random.uniform(-m, m)]])
            f = cv2.warpAffine(f, M, (W, H))
        if self.flash > 0:
            f = cv2.addWeighted(f, 1 - self.flash, np.full_like(f, (0, 0, 255)), self.flash, 0)
        if self.pinch_active:  # blue-ish tint for slow-mo
            f = cv2.addWeighted(f, 0.9, np.full_like(f, (120, 50, 30)), 0.1, 0)
        return f

    def draw_hud(self, f, fps):
        put_text(f, f"SCORE {self.score}", (24, 44), 1.1, (255, 255, 255), 2)
        put_text(f, f"BEST {self.best}", (24, 78), 0.65, (0, 215, 255), 1)
        put_text(f, f"LEVEL {self.level}", (W // 2 - 90, 40), 0.9, (0, 255, 200), 2)
        mult = self.multiplier()
        ccol = (0, 215, 255) if mult > 1 else (200, 200, 200)
        put_text(f, f"COMBO {self.combo}  x{mult}", (W // 2 - 130, 76), 0.75, ccol, 2)

        # lives
        for i in range(MAX_LIVES):
            c = (W - 36 - i * 38, 36)
            if i < self.lives:
                cv2.circle(f, c, 13, (180, 105, 255), -1, cv2.LINE_AA)
            cv2.circle(f, c, 13, (255, 255, 255), 2, cv2.LINE_AA)
        put_text(f, f"Misses {self.misses}/5", (W - 160, 76), 0.6, (200, 200, 200), 1)

        # energy bar
        bx, by, bw, bh = 24, 100, 240, 16
        cv2.rectangle(f, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(f, (bx, by), (bx + int(bw * self.energy / 100), by + bh), (230, 90, 190), -1)
        cv2.rectangle(f, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        put_text(f, "SLOW-MO ENERGY (pinch)", (bx, by + 38), 0.5, (230, 170, 220), 1)

        # active power-ups
        y = by + 66
        for label, on, col in (
            (f"SHIELD READY", self.shield, (255, 150, 50)),
            (f"SLOW-MO {self.slow_t:.0f}s", self.slow_t > 0, (230, 90, 190)),
            (f"2x POINTS {self.double_t:.0f}s", self.double_t > 0, (0, 140, 255)),
        ):
            if on:
                put_text(f, label, (bx, y), 0.6, col, 2)
                y += 28
        put_text(f, f"{fps:.0f} FPS", (W - 110, H - 12 - (TILE_H + MARGIN if True else 0)), 0.45,
                 (150, 150, 150), 1)


# ----------------------------------------------------------------- Zoom-style camera tile
class CameraTile:
    def __init__(self):
        self.outer = rounded_mask(TILE_W, TILE_H, 16)
        self.iw, self.ih = TILE_W - 6, TILE_H - 6
        self.inner = rounded_mask(self.iw, self.ih, 13)

    def draw(self, canvas, cam, tracker):
        x0, y0 = W - TILE_W - MARGIN, H - TILE_H - MARGIN
        # soft shadow
        s = canvas[y0 - 8:y0 + TILE_H + 8, x0 - 8:x0 + TILE_W + 8]
        s[:] = (s * 0.5).astype(np.uint8)

        # centre-crop to 4:3
        h, w = cam.shape[:2]
        target_w = int(h * 4 / 3)
        if target_w < w:
            off = (w - target_w) // 2
            cam = cam[:, off:off + target_w]
        tile = cv2.resize(cam, (self.iw, self.ih))

        # name bar
        bar = tile[-34:]
        bar[:] = (bar * 0.35).astype(np.uint8)
        gest = tracker.gesture.upper() if tracker.present else "NO HAND"
        put_text(tile, "You", (34, self.ih - 11), 0.55, (255, 255, 255), 1)
        put_text(tile, gest, (self.iw - 100, self.ih - 11), 0.5, (0, 255, 180), 1)
        cv2.circle(tile, (18, self.ih - 17), 6, (80, 80, 255) if not tracker.present else (0, 220, 120), -1)

        # LIVE badge
        cv2.rectangle(tile, (8, 8), (62, 28), (40, 40, 220), -1)
        put_text(tile, "LIVE", (14, 23), 0.45, (255, 255, 255), 1, )

        border = (0, 220, 120) if tracker.present else (110, 110, 110)
        roi = canvas[y0:y0 + TILE_H, x0:x0 + TILE_W]
        roi[self.outer] = border
        sub = roi[3:3 + self.ih, 3:3 + self.iw]
        sub[self.inner] = tile[self.inner]


# ----------------------------------------------------------------- screens
def hold_ring(f, center, progress):
    cv2.ellipse(f, center, (34, 34), -90, 0, 360, (80, 80, 80), 5, cv2.LINE_AA)
    cv2.ellipse(f, center, (34, 34), -90, 0, int(360 * min(1, progress)), (0, 255, 180), 5, cv2.LINE_AA)


def draw_overlay(f, state, game, hold_prog):
    if state == "playing":
        put_text(f, "Point: move  |  Pinch: slow-mo  |  Fist (hold): pause", (24, H - 20), 0.5,
                 (190, 190, 190), 1)
        return
    f[:] = dim(f, 0.6)
    cx, cy = W // 2, H // 2
    if state == "menu":
        put_text(f, "HAND RUSH", (cx, cy - 170), 3.0, (255, 220, 80), 5, center=True)
        put_text(f, "Catch the orbs. Dodge the bombs. Use only your hand.", (cx, cy - 100), 0.9,
                 (230, 230, 230), 1, center=True)
        lines = ["POINT  - move the catcher", "PINCH  - slow-mo (needs energy)",
                 "FIST   - pause (hold)", "PALM   - start / resume (hold 1s)"]
        for i, s in enumerate(lines):
            put_text(f, s, (cx - 190, cy - 30 + i * 34), 0.75, (255, 255, 255), 1)
        put_text(f, "Hold an OPEN PALM to start  (or press SPACE)", (cx, cy + 150), 0.95,
                 (0, 255, 180), 2, center=True)
        put_text(f, f"Best score: {game.best}", (cx, cy + 195), 0.8, (0, 215, 255), 1, center=True)
    elif state == "paused":
        put_text(f, "PAUSED", (cx, cy - 40), 2.6, (255, 255, 255), 4, center=True)
        put_text(f, "Hold an OPEN PALM to resume  (or press SPACE)", (cx, cy + 50), 0.9,
                 (0, 255, 180), 2, center=True)
    elif state == "over":
        put_text(f, "GAME OVER", (cx, cy - 150), 2.6, (80, 80, 255), 4, center=True)
        put_text(f, f"Score {game.score}", (cx, cy - 70), 1.4, (255, 255, 255), 2, center=True)
        put_text(f, f"Level {game.level}   Caught {game.caught}   Best combo {game.best_combo}",
                 (cx, cy - 20), 0.8, (210, 210, 210), 1, center=True)
        if game.new_record:
            put_text(f, "NEW RECORD!", (cx, cy + 30), 1.2, (0, 215, 255), 3, center=True)
        else:
            put_text(f, f"Best {game.best}", (cx, cy + 30), 0.9, (0, 215, 255), 1, center=True)
        put_text(f, "Hold an OPEN PALM to play again  (or press R)", (cx, cy + 100), 0.9,
                 (0, 255, 180), 2, center=True)
    if hold_prog > 0:
        hold_ring(f, (cx, cy + 250), hold_prog)


# ----------------------------------------------------------------- main loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera. Close other apps using it or try VideoCapture(1).")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker, game, tile = HandTracker(), Game(), CameraTile()
    win = "HandRush"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    state = "menu"          # menu | playing | paused | over
    show_cam, fullscreen = True, False
    hold, prev, fps = 0.0, time.perf_counter(), 30.0

    while True:
        ok, cam = cap.read()
        if not ok:
            break
        cam = cv2.flip(cam, 1)      # mirror, like a video call
        tracker.update(cam)

        now = time.perf_counter()
        dt = min(0.05, now - prev)
        prev = now
        fps = fps * 0.9 + (1 / max(dt, 1e-3)) * 0.1

        # ---- gesture-driven state machine
        want = "fist" if state == "playing" else "palm"
        need = 0.7 if state == "playing" else 1.0
        hold = hold + dt if (tracker.present and tracker.gesture == want) else 0.0
        if hold >= need:
            hold = 0.0
            if state == "playing":
                state = "paused"
            else:
                if state in ("menu", "over"):
                    game.reset()
                state = "playing"

        if state == "playing":
            if not tracker.present and now - tracker.last_seen > 2.0:
                state = "paused"    # auto-pause when your hand disappears
            else:
                game.update(dt, tracker.pos, tracker.gesture == "pinch" and tracker.present)
                if game.over:
                    state = "over"

        # ---- render
        frame = game.draw_world(tracker)
        if state in ("playing", "paused", "over"):
            game.draw_hud(frame, fps)
        draw_overlay(frame, state, game, hold / need)
        if not tracker.present and state == "playing":
            put_text(frame, "Show your hand to the camera", (W // 2, H - 60), 0.9, (80, 180, 255), 2,
                     center=True)
        if show_cam:
            tile.draw(frame, cam, tracker)
        cv2.imshow(win, frame)

        # ---- keyboard fallbacks
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            if state in ("menu", "over"):
                game.reset()
                state = "playing"
            elif state == "playing":
                state = "paused"
            else:
                state = "playing"
        elif key == ord("r"):
            game.reset()
            state = "playing"
        elif key == ord("c"):
            show_cam = not show_cam
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()