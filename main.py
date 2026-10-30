"""
HandRush - a hand-gesture arcade game, Vercel edition  (single file, zero dependencies)

WHY THE OLD VERSION FAILED ON VERCEL
  Vercel runs your code on a serverless machine: no webcam, no screen (cv2.imshow),
  no writable disk, and opencv + mediapipe are far bigger than the 250 MB limit.
  So the game now runs in the visitor's BROWSER (canvas + MediaPipe Tasks Vision JS,
  same gestures, same rules) and this Python file only serves that page.

DEPLOY
  1. Keep this file as  app.py  in the repo root (index.py also works).
  2. Delete any requirements.txt that lists opencv / mediapipe / numpy
     (nothing needs installing - this file uses only the Python standard library).
  3. `vercel` (or push to a repo connected to Vercel). Open the https:// URL.

RUN LOCALLY
  python app.py      ->  http://localhost:8000

GESTURES
  Point (index finger) ... move the catcher
  Pinch (thumb + index) .. slow-motion (drains the energy bar)
  Open palm (hold 1s) .... start / restart / resume
  Fist (hold 0.7s) ....... pause

KEYS / MOUSE
  SPACE start/pause   R restart   C hide/show camera tile   F fullscreen
  No camera?  Choose "Play with mouse / touch": move = catcher, hold click = slow-mo.
"""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>HandRush</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; height: 100%; background: #05050e; overflow: hidden;
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; color: #fff; }
  body { display: flex; align-items: center; justify-content: center; }
  canvas { display: block; background: #140a28; touch-action: none; }
  #gate { position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 14px; padding: 24px; text-align: center;
    background: rgba(8, 4, 20, .86); }
  #gate[hidden] { display: none; }
  #gate h1 { margin: 0; font-size: clamp(40px, 9vw, 84px); letter-spacing: .04em; color: #ffdc50; }
  #gate p { margin: 0; max-width: 46ch; color: #d8d8e4; line-height: 1.45; }
  #gate small { color: #8d8da6; max-width: 52ch; }
  button { font: inherit; font-weight: 600; font-size: 18px; padding: 14px 28px; border-radius: 12px;
    border: 2px solid #00ffb4; background: #00ffb4; color: #04140e; cursor: pointer; }
  button:hover { filter: brightness(1.1); }
  button:focus-visible { outline: 3px solid #fff; outline-offset: 3px; }
  button[disabled] { opacity: .5; cursor: progress; }
  button.ghost { background: transparent; color: #00ffb4; }
  #status { min-height: 1.4em; color: #ffb45a; }
  video { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
</style>
</head>
<body>
<canvas id="c" width="1280" height="720"></canvas>
<video id="cam" playsinline muted></video>

<div id="gate">
  <h1>HAND RUSH</h1>
  <p>Catch the orbs, dodge the bombs, and play with nothing but your hand in front of the webcam.</p>
  <button id="btnCam">Enable camera</button>
  <button id="btnMouse" class="ghost">Play with mouse / touch</button>
  <p id="status" role="status"></p>
  <small>Hand tracking runs inside your browser. Your video is never uploaded or stored.</small>
</div>

<script type="module">
"use strict";

// ------------------------------------------------------------- config
const W = 1280, H = 720;
const TILE_W = 320, TILE_H = 240, MARGIN = 24;
const CATCHER_R = 46, MAX_LIVES = 5;
const FONT = '"Segoe UI", system-ui, -apple-system, Roboto, sans-serif';
const VERSION = "0.10.14";
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// colours are [r, g, b]
const KINDS = {
  orb:    { w: 58, r: 26, col: [60, 200, 255],  label: "" },
  gold:   { w: 10, r: 22, col: [255, 215, 0],   label: "$" },
  bomb:   { w: 18, r: 28, col: [55, 45, 45],    label: "" },
  heart:  { w: 3,  r: 24, col: [255, 105, 180], label: "+" },
  shield: { w: 4,  r: 24, col: [50, 150, 255],  label: "S" },
  slow:   { w: 4,  r: 24, col: [190, 90, 230],  label: "T" },
  double: { w: 4,  r: 24, col: [255, 140, 0],   label: "2x" },
};
const RED = [255, 40, 40], WHITE = [255, 255, 255], TEAL = [0, 255, 180], GOLD = [255, 215, 0];
const HAND_CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const gate = document.getElementById("gate");
const statusEl = document.getElementById("status");
const btnCam = document.getElementById("btnCam");
const btnMouse = document.getElementById("btnMouse");

// ------------------------------------------------------------- helpers
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const rand = (a, b) => a + Math.random() * (b - a);
const rgba = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

function text(s, x, y, scale = 1, col = WHITE, thick = 2, center = false) {
  ctx.font = `${thick >= 3 ? 800 : 600} ${Math.round(scale * 32)}px ${FONT}`;
  ctx.textAlign = center ? "center" : "left";
  ctx.textBaseline = center ? "middle" : "alphabetic";
  ctx.lineJoin = "round";
  ctx.lineWidth = thick + 3;
  ctx.strokeStyle = "rgba(0,0,0,.75)";
  ctx.strokeText(s, x, y);
  ctx.fillStyle = rgba(col);
  ctx.fillText(s, x, y);
}

function circle(x, y, r, fill, stroke, lw = 2) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.lineWidth = lw; ctx.strokeStyle = stroke; ctx.stroke(); }
}

function line(x1, y1, x2, y2, col, lw) {
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.lineCap = "round"; ctx.stroke();
}

function rr(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function loadBest() {
  try { return parseInt(localStorage.getItem("handrush_best") || "0", 10) || 0; } catch { return 0; }
}
function saveBest(v) {
  try { localStorage.setItem("handrush_best", String(v)); } catch { /* storage blocked */ }
}

// ------------------------------------------------------------- hand tracking
let landmarker = null;
let lastVideoTime = -1;
let mode = "none"; // "hand" | "mouse" | "none"

const tracker = {
  pos: { x: W / 2, y: H / 2 }, present: false, lastSeen: performance.now() / 1000,
  gesture: "none", pinchRatio: 1, lm: null,
};

const dist2 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function classify(p) {
  let ext = 0;
  for (const [t, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) {
    if (dist2(p[t], p[0]) > dist2(p[pip], p[0]) * 1.05) ext++;
  }
  const scale = dist2(p[0], p[9]) || 1e-6;
  tracker.pinchRatio = dist2(p[4], p[8]) / scale;
  if (tracker.pinchRatio < 0.28) return "pinch";
  if (ext >= 4) return "palm";
  if (ext === 0) return "fist";
  return "point";
}

function trackHand(now) {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  let res;
  try { res = landmarker.detectForVideo(video, performance.now()); } catch { return; }
  if (res.landmarks && res.landmarks.length) {
    const p = res.landmarks[0];
    tracker.lm = p;
    tracker.present = true;
    tracker.lastSeen = now;
    const mx = 1 - p[8].x; // mirror, like a video call
    const tx = clamp((mx - 0.12) / 0.76, 0, 1) * W;
    const ty = clamp((p[8].y - 0.10) / 0.70, 0, 1) * H;
    const d = Math.hypot(tx - tracker.pos.x, ty - tracker.pos.y);
    const a = clamp(d / 160, 0.25, 0.85); // adaptive smoothing
    tracker.pos.x += (tx - tracker.pos.x) * a;
    tracker.pos.y += (ty - tracker.pos.y) * a;
    tracker.gesture = classify(p);
  } else {
    tracker.present = false;
    tracker.gesture = "none";
    tracker.lm = null;
  }
}

// mouse / touch fallback
const mouse = { down: false };
function pointerToStage(e) {
  const r = canvas.getBoundingClientRect();
  tracker.pos.x = clamp((e.clientX - r.left) / r.width * W, 0, W);
  tracker.pos.y = clamp((e.clientY - r.top) / r.height * H, 0, H);
}
canvas.addEventListener("pointermove", (e) => { if (mode === "mouse") pointerToStage(e); });
canvas.addEventListener("pointerdown", (e) => {
  if (mode === "mouse") pointerToStage(e);
  if (state !== "playing") { startOrResume(); return; }
  if (mode === "mouse") mouse.down = true;
});
window.addEventListener("pointerup", () => { mouse.down = false; });
window.addEventListener("pointercancel", () => { mouse.down = false; });

// ------------------------------------------------------------- game logic
class Game {
  constructor() {
    this.best = loadBest();
    this.stars = Array.from({ length: 90 }, () => [rand(0, W), rand(0, H), rand(15, 70), 1 + Math.floor(Math.random() * 2)]);
    this.reset();
  }

  reset() {
    this.score = 0; this.lives = 3; this.level = 1; this.combo = 0; this.bestCombo = 0;
    this.caught = 0; this.misses = 0; this.energy = 0;
    this.shield = false; this.slowT = 0; this.doubleT = 0;
    this.items = []; this.particles = []; this.popups = []; this.trail = [];
    this.spawnT = 1.0; this.t = 0; this.flash = 0; this.shake = 0;
    this.pinchActive = false; this.over = false; this.newRecord = false;
  }

  multiplier() { return Math.min(5, 1 + Math.floor(this.combo / 5)); }

  burst(x, y, col, n = 18) {
    for (let i = 0; i < n; i++) {
      const a = rand(0, Math.PI * 2), sp = rand(80, 340), life = rand(0.4, 0.9);
      this.particles.push([x, y, Math.cos(a) * sp, Math.sin(a) * sp, life, life, col, 2 + Math.floor(Math.random() * 4)]);
    }
  }

  popup(x, y, s, col = WHITE, size = 0.9) { this.popups.push([x, y, s, 1.0, col, size]); }

  loseLife() {
    this.lives -= 1;
    this.flash = 0.5; this.shake = 0.4; this.combo = 0;
    if (this.lives <= 0) {
      this.over = true;
      if (this.score > this.best) { this.best = this.score; this.newRecord = true; saveBest(this.best); }
    }
  }

  spawn() {
    const kinds = Object.keys(KINDS);
    const weights = kinds.map((k) => KINDS[k].w + (k === "bomb" ? this.level * 1.5 : 0));
    let roll = Math.random() * weights.reduce((a, b) => a + b, 0), k = kinds[0];
    for (let i = 0; i < kinds.length; i++) { roll -= weights[i]; if (roll <= 0) { k = kinds[i]; break; } }
    const r = KINDS[k].r;
    this.items.push({ k, x: rand(r + 20, W - r - 20), y: -r, vy: 210 + this.level * 38 + rand(-30, 60), r, rot: rand(0, 6) });
  }

  update(dt, cur, pinching) {
    this.t += dt;
    let scale = 1.0;
    if (this.slowT > 0) { this.slowT -= dt; scale = 0.5; }
    this.pinchActive = !!(pinching && this.energy > 0);
    if (this.pinchActive) { this.energy = Math.max(0, this.energy - 28 * dt); scale = Math.min(scale, 0.4); }
    this.doubleT = Math.max(0, this.doubleT - dt);
    this.flash = Math.max(0, this.flash - dt);
    this.shake = Math.max(0, this.shake - dt);
    const sdt = dt * scale;

    this.trail.push([cur.x, cur.y]);
    if (this.trail.length > 14) this.trail.shift();

    this.spawnT -= sdt;
    if (this.spawnT <= 0) {
      this.spawn();
      this.spawnT = Math.max(0.22, 0.85 - 0.06 * (this.level - 1)) * rand(0.7, 1.3);
    }

    const alive = [];
    for (const it of this.items) {
      it.y += it.vy * sdt;
      it.rot += 2.5 * sdt;
      if (Math.hypot(it.x - cur.x, it.y - cur.y) < it.r + CATCHER_R) { this.onCatch(it); continue; }
      if (it.y > H + 40) {
        if (it.k === "orb" || it.k === "gold") {
          this.misses++; this.combo = 0;
          if (this.misses >= 5) {
            this.misses = 0;
            this.popup(it.x, H - 60, "5 MISSES!", [255, 80, 80]);
            this.loseLife();
          }
        }
        continue;
      }
      alive.push(it);
    }
    this.items = alive;

    for (const p of this.particles) {
      p[0] += p[2] * dt; p[1] += p[3] * dt; p[3] += 400 * dt; p[4] -= dt;
    }
    this.particles = this.particles.filter((p) => p[4] > 0);
    for (const pu of this.popups) { pu[1] -= 60 * dt; pu[3] -= dt * 0.9; }
    this.popups = this.popups.filter((pu) => pu[3] > 0);
    for (const s of this.stars) {
      s[1] += s[2] * dt * (scale < 1 ? 0.5 : 1.0);
      if (s[1] > H) { s[0] = rand(0, W); s[1] = 0; }
    }
  }

  onCatch(it) {
    const { k, x, y } = it, col = KINDS[k].col;
    if (k === "bomb") {
      if (this.shield) {
        this.shield = false;
        this.burst(x, y, [50, 150, 255], 30);
        this.popup(x, y, "BLOCKED", [80, 180, 255]);
      } else {
        this.burst(x, y, RED, 36);
        this.popup(x, y, "BOOM!", [255, 60, 60], 1.2);
        this.loseLife();
      }
      return;
    }
    this.burst(x, y, col);
    if (k === "orb" || k === "gold") {
      const base = k === "orb" ? 10 : 30;
      this.combo++;
      this.bestCombo = Math.max(this.bestCombo, this.combo);
      this.caught++;
      const gained = base * this.multiplier() * (this.doubleT > 0 ? 2 : 1);
      this.score += gained;
      this.energy = Math.min(100, this.energy + (k === "orb" ? 4 : 10));
      this.popup(x, y, `+${gained}`, col);
      const newLevel = 1 + Math.floor(this.score / 200);
      if (newLevel > this.level) {
        this.level = newLevel;
        this.popup(W / 2, H / 2 - 80, `LEVEL ${this.level}`, [0, 255, 200], 1.8);
      }
    } else if (k === "heart") {
      this.lives = Math.min(MAX_LIVES, this.lives + 1);
      this.popup(x, y, "+1 LIFE", col);
    } else if (k === "shield") {
      this.shield = true; this.popup(x, y, "SHIELD", col);
    } else if (k === "slow") {
      this.slowT = 5.0; this.popup(x, y, "SLOW-MO", col);
    } else if (k === "double") {
      this.doubleT = 8.0; this.popup(x, y, "2x POINTS", col);
    }
  }

  // ---- drawing
  drawItem(it) {
    const k = KINDS[it.k], x = it.x, y = it.y, r = it.r;
    if (it.k === "bomb") {
      circle(x, y, r, rgba(k.col), rgba(RED), 3);
      for (const a of [it.rot, it.rot + Math.PI / 2]) {
        const dx = Math.cos(a) * r * 0.55, dy = Math.sin(a) * r * 0.55;
        line(x - dx, y - dy, x + dx, y + dy, rgba([255, 60, 60]), 3);
      }
    } else {
      circle(x, y, r, rgba(k.col), "#fff", 2);
      circle(x - r / 3, y - r / 3, Math.max(2, r / 5), "#fff");
      if (k.label) text(k.label, x + 2, y + 2, 0.65, WHITE, 2, true);
    }
  }

  drawWorld() {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "rgb(20,20,48)");
    g.addColorStop(1, "rgb(40,6,22)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    if (this.shake > 0) {
      const m = this.shake * 30;
      ctx.translate(rand(-m, m), rand(-m, m));
    }

    for (const s of this.stars) {
      const c = 120 + s[3] * 40;
      circle(s[0], s[1], s[3], `rgb(${c},${c},${c})`);
    }

    // glow
    ctx.globalAlpha = 0.28;
    for (const it of this.items) {
      circle(it.x, it.y, it.r * 1.8, rgba(it.k === "bomb" ? RED : KINDS[it.k].col));
    }
    ctx.globalAlpha = 1;
    for (const it of this.items) this.drawItem(it);

    // trail
    const pts = this.trail;
    for (let i = 1; i < pts.length; i++) {
      const a = i / pts.length;
      line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1],
        `rgb(${90 * a | 0},${220 * a | 0},${255 * a | 0})`, Math.max(1, 14 * a));
    }

    for (const p of this.particles) circle(p[0], p[1], p[7], rgba(p[6], p[4] / p[5]));

    // catcher
    const cx = tracker.pos.x, cy = tracker.pos.y;
    let ring = this.pinchActive ? [190, 90, 230] : [120, 255, 255];
    if (!tracker.present) ring = [120, 120, 120];
    circle(cx, cy, CATCHER_R, null, rgba(ring), 3);
    circle(cx, cy, 6, rgba(ring));
    if (this.shield) {
      const pulse = Math.floor(8 * Math.sin(this.t * 6));
      circle(cx, cy, CATCHER_R + 14 + Math.trunc(pulse / 2), null, rgba([50, 150, 255]), 3);
    }

    for (const [x, y, s, life, col, size] of this.popups) {
      ctx.globalAlpha = Math.min(1, life * 1.5);
      text(s, x, y, size, col, 2, true);
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    if (this.flash > 0) { ctx.fillStyle = rgba(RED, this.flash); ctx.fillRect(0, 0, W, H); }
    if (this.pinchActive) { ctx.fillStyle = "rgba(30,50,120,.1)"; ctx.fillRect(0, 0, W, H); }
  }

  drawHud(fps) {
    text(`SCORE ${this.score}`, 24, 44, 1.1, WHITE, 2);
    text(`BEST ${this.best}`, 24, 78, 0.65, GOLD, 1);
    text(`LEVEL ${this.level}`, W / 2 - 90, 40, 0.9, [0, 255, 200], 2);
    const mult = this.multiplier();
    text(`COMBO ${this.combo}  x${mult}`, W / 2 - 130, 76, 0.75, mult > 1 ? GOLD : [200, 200, 200], 2);

    for (let i = 0; i < MAX_LIVES; i++) {
      const cx = W - 36 - i * 38, cy = 36;
      circle(cx, cy, 13, i < this.lives ? "rgb(255,105,180)" : null, "#fff", 2);
    }
    text(`Misses ${this.misses}/5`, W - 160, 76, 0.6, [200, 200, 200], 1);

    const bx = 24, by = 100, bw = 240, bh = 16;
    ctx.fillStyle = "#3c3c3c"; ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = "rgb(190,90,230)"; ctx.fillRect(bx, by, bw * this.energy / 100, bh);
    ctx.lineWidth = 1; ctx.strokeStyle = "#fff"; ctx.strokeRect(bx, by, bw, bh);
    text(mode === "mouse" ? "SLOW-MO ENERGY (hold click)" : "SLOW-MO ENERGY (pinch)", bx, by + 38, 0.5, [220, 170, 230], 1);

    let y = by + 66;
    for (const [label, on, col] of [
      ["SHIELD READY", this.shield, [50, 150, 255]],
      [`SLOW-MO ${Math.ceil(this.slowT)}s`, this.slowT > 0, [190, 90, 230]],
      [`2x POINTS ${Math.ceil(this.doubleT)}s`, this.doubleT > 0, [255, 140, 0]],
    ]) {
      if (on) { text(label, bx, y, 0.6, col, 2); y += 28; }
    }
    text(`${fps.toFixed(0)} FPS`, W - 110, H - 12 - (mode === "hand" ? TILE_H + MARGIN : 0), 0.45, [150, 150, 150], 1);
  }
}

// ------------------------------------------------------------- Zoom-style camera tile
function drawTile() {
  const x0 = W - TILE_W - MARGIN, y0 = H - TILE_H - MARGIN, iw = TILE_W - 6, ih = TILE_H - 6;
  ctx.save();
  rr(x0 - 8, y0 - 8, TILE_W + 16, TILE_H + 16, 22); ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fill();
  rr(x0, y0, TILE_W, TILE_H, 16); ctx.fillStyle = tracker.present ? "#00dc78" : "#6e6e6e"; ctx.fill();

  rr(x0 + 3, y0 + 3, iw, ih, 13); ctx.clip();
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  const sw = Math.min(vw, vh * 4 / 3), sx = (vw - sw) / 2; // centre-crop to 4:3
  if (video.readyState >= 2) {
    ctx.save();
    ctx.translate(x0 + 3 + iw, y0 + 3);
    ctx.scale(-1, 1); // mirror
    ctx.drawImage(video, sx, 0, sw, vh, 0, 0, iw, ih);
    ctx.restore();
  } else {
    ctx.fillStyle = "#111"; ctx.fillRect(x0 + 3, y0 + 3, iw, ih);
  }

  // skeleton
  if (tracker.lm) {
    const P = tracker.lm.map((l) => [x0 + 3 + iw - ((l.x * vw - sx) / sw) * iw, y0 + 3 + l.y * ih]);
    for (const [a, b] of HAND_CONNECTIONS) line(P[a][0], P[a][1], P[b][0], P[b][1], "#00ffb4", 3);
    for (const p of P) circle(p[0], p[1], 3.5, "#fff");
  }

  // name bar
  ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(x0 + 3, y0 + 3 + ih - 34, iw, 34);
  text("You", x0 + 3 + 34, y0 + 3 + ih - 11, 0.55, WHITE, 1);
  const gest = tracker.present ? tracker.gesture.toUpperCase() : "NO HAND";
  ctx.font = `600 16px ${FONT}`;
  const gw = ctx.measureText(gest).width;
  text(gest, x0 + 3 + iw - gw - 10, y0 + 3 + ih - 11, 0.5, TEAL, 1);
  circle(x0 + 3 + 18, y0 + 3 + ih - 17, 6, tracker.present ? "rgb(0,220,120)" : "rgb(255,80,80)");

  // LIVE badge
  ctx.fillStyle = "rgb(220,40,40)"; ctx.fillRect(x0 + 3 + 8, y0 + 3 + 8, 54, 20);
  text("LIVE", x0 + 3 + 14, y0 + 3 + 23, 0.45, WHITE, 1);
  ctx.restore();
}

// ------------------------------------------------------------- screens
function holdRing(cx, cy, progress) {
  ctx.lineWidth = 5;
  ctx.beginPath(); ctx.arc(cx, cy, 34, 0, Math.PI * 2); ctx.strokeStyle = "#505050"; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
  ctx.strokeStyle = "#00ffb4"; ctx.stroke();
}

function drawOverlay(state, game, holdProg) {
  const m = mode === "mouse";
  if (state === "playing") {
    text(m ? "Mouse: move  |  Hold click: slow-mo  |  SPACE: pause"
           : "Point: move  |  Pinch: slow-mo  |  Fist (hold): pause", 24, H - 20, 0.5, [190, 190, 190], 1);
    return;
  }
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2;
  if (state === "menu") {
    text("HAND RUSH", cx, cy - 170, 3.0, [255, 220, 80], 5, true);
    text("Catch the orbs. Dodge the bombs. Use only your hand.", cx, cy - 100, 0.9, [230, 230, 230], 1, true);
    const lines = m
      ? ["MOUSE  - move the catcher", "CLICK  - hold for slow-mo (needs energy)", "SPACE  - pause / resume", "TAP    - start / resume"]
      : ["POINT  - move the catcher", "PINCH  - slow-mo (needs energy)", "FIST   - pause (hold)", "PALM   - start / resume (hold 1s)"];
    lines.forEach((s, i) => text(s, cx - 230, cy - 30 + i * 34, 0.75, WHITE, 1));
    text(m ? "Click / tap or press SPACE to start" : "Hold an OPEN PALM to start  (or press SPACE)", cx, cy + 150, 0.95, TEAL, 2, true);
    text(`Best score: ${game.best}`, cx, cy + 195, 0.8, GOLD, 1, true);
  } else if (state === "paused") {
    text("PAUSED", cx, cy - 40, 2.6, WHITE, 4, true);
    text(m ? "Click / tap or press SPACE to resume" : "Hold an OPEN PALM to resume  (or press SPACE)", cx, cy + 50, 0.9, TEAL, 2, true);
  } else if (state === "over") {
    text("GAME OVER", cx, cy - 150, 2.6, [255, 80, 80], 4, true);
    text(`Score ${game.score}`, cx, cy - 70, 1.4, WHITE, 2, true);
    text(`Level ${game.level}   Caught ${game.caught}   Best combo ${game.bestCombo}`, cx, cy - 20, 0.8, [210, 210, 210], 1, true);
    if (game.newRecord) text("NEW RECORD!", cx, cy + 30, 1.2, GOLD, 3, true);
    else text(`Best ${game.best}`, cx, cy + 30, 0.9, GOLD, 1, true);
    text(m ? "Click / tap or press R to play again" : "Hold an OPEN PALM to play again  (or press R)", cx, cy + 100, 0.9, TEAL, 2, true);
  }
  if (holdProg > 0) holdRing(cx, cy + 250, holdProg);
}

// ------------------------------------------------------------- main loop
const game = new Game();
let state = "menu"; // menu | playing | paused | over
let showCam = true;
let hold = 0, prev = performance.now() / 1000, fps = 60;

function startOrResume() {
  if (mode === "none") return;          // wait for the gate choice
  if (state === "menu" || state === "over") game.reset();
  state = "playing";
  mouse.down = false;
}

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now() / 1000;
  const dt = Math.min(0.05, now - prev);
  prev = now;
  fps = fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

  if (mode === "hand") {
    trackHand(now);
  } else if (mode === "mouse") {
    tracker.present = true;
    tracker.lastSeen = now;
    tracker.gesture = mouse.down ? "pinch" : "point";
  }

  // gesture-driven state machine
  const want = state === "playing" ? "fist" : "palm";
  const need = state === "playing" ? 0.7 : 1.0;
  hold = (mode === "hand" && tracker.present && tracker.gesture === want) ? hold + dt : 0;
  if (hold >= need) {
    hold = 0;
    if (state === "playing") state = "paused"; else startOrResume();
  }

  if (state === "playing") {
    if (mode === "hand" && !tracker.present && now - tracker.lastSeen > 2.0) {
      state = "paused"; // auto-pause when the hand disappears
    } else {
      game.update(dt, tracker.pos, tracker.gesture === "pinch" && tracker.present);
      if (game.over) state = "over";
    }
  }

  // render
  game.drawWorld();
  if (state !== "menu") game.drawHud(fps);
  drawOverlay(state, game, hold / need);
  if (mode === "hand" && !tracker.present && state === "playing") {
    text("Show your hand to the camera", W / 2, H - 60, 0.9, [255, 180, 80], 2, true);
  }
  if (mode === "hand" && showCam) drawTile();
}

// ------------------------------------------------------------- input: keyboard
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === " ") {
    e.preventDefault();
    if (mode === "none") return;
    if (state === "playing") state = "paused"; else startOrResume();
  } else if (k === "r") {
    if (mode === "none") return;
    game.reset(); state = "playing";
  } else if (k === "c") {
    showCam = !showCam;
  } else if (k === "f") {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  } else if ((k === "escape" || k === "q") && state === "playing") {
    state = "paused";
  }
});
document.addEventListener("visibilitychange", () => { if (document.hidden && state === "playing") state = "paused"; });

// ------------------------------------------------------------- layout
function fit() {
  const s = Math.min(window.innerWidth / W, window.innerHeight / H);
  canvas.style.width = Math.floor(W * s) + "px";
  canvas.style.height = Math.floor(H * s) + "px";
}
window.addEventListener("resize", fit);
fit();

// ------------------------------------------------------------- start-up (camera or mouse)
const setStatus = (s) => { statusEl.textContent = s; };

async function startCamera() {
  btnCam.disabled = true; btnMouse.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser can't access a camera here (HTTPS is required).");
    }
    setStatus("Loading hand-tracking model…");
    const vision = await import(BUNDLE_URL);
    const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
    const make = (delegate) => vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    try { landmarker = await make("GPU"); } catch { landmarker = await make("CPU"); }

    setStatus("Waiting for camera permission…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    mode = "hand";
    gate.hidden = true;
  } catch (err) {
    const name = err && err.name;
    if (name === "NotAllowedError") setStatus("Camera permission was blocked. Allow it in the address bar, then try again.");
    else if (name === "NotFoundError") setStatus("No camera found. Plug one in, or play with the mouse.");
    else if (name === "NotReadableError") setStatus("The camera is busy in another app. Close it and try again.");
    else setStatus((err && err.message) || "Could not start the camera or load the model.");
    btnCam.disabled = false; btnMouse.disabled = false;
    btnCam.textContent = "Try camera again";
  }
}

btnCam.addEventListener("click", startCamera);
btnMouse.addEventListener("click", () => { mode = "mouse"; gate.hidden = true; });

requestAnimationFrame(frame);
</script>
</body>
</html>
"""

BODY = PAGE.encode("utf-8")
HEADERS = [
    ("Content-Type", "text/html; charset=utf-8"),
    ("Cache-Control", "public, max-age=0, must-revalidate"),
    ("Permissions-Policy", "camera=(self)"),
    ("X-Content-Type-Options", "nosniff"),
]


def _respond(path):
    """Returns (status_code, headers, body) for a request path."""
    if path == "/favicon.ico":
        return 204, [], b""
    return 200, HEADERS + [("Content-Length", str(len(BODY)))], BODY


# --- Vercel entrypoint #1: WSGI callable named `app` ------------------------
def app(environ, start_response):
    code, headers, body = _respond(environ.get("PATH_INFO", "/"))
    start_response(f"{code} {HTTPStatus(code).phrase}", headers)
    return [body]


# --- Vercel entrypoint #2: classic `handler` class (works in /api too) ------
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        code, headers, body = _respond(self.path.split("?", 1)[0])
        self.send_response(code)
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep logs quiet
        pass


# --- Local run: python app.py ----------------------------------------------
if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    print("HandRush running at http://localhost:8000  (Ctrl+C to stop)")
    make_server("", 8000, app).serve_forever()
