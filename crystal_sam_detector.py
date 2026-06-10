"""
Real-time crystal detector — SAM3 Interactive GUI.

Draw a box around ONE reference crystal on the first frame.
SAM3 then finds and counts every matching crystal in every subsequent frame.

Usage:
    python crystal_sam_detector.py                        # live camera (auto-detected)
    python crystal_sam_detector.py --camera 1             # specific camera index
    python crystal_sam_detector.py --video path.mp4       # run on a video file
    python crystal_sam_detector.py --ref ref.png          # separate reference image

Controls (setup screen):
    Left-click + drag   → draw ROI box around reference crystal
    ENTER / SPACE       → confirm box and start detection
    r                   → redraw box
    q / ESC             → quit

Controls (detection screen):
    SPACE               → pause / resume
    r                   → re-draw reference box (restart setup)
    s                   → save current frame as PNG
    q / ESC             → quit

Checkpoint:
    Put sam3.pt in this folder, or set --checkpoint path.
    If not found locally, the model is loaded from HuggingFace automatically.

Dependencies:
    pip install opencv-python numpy torch torchvision
    (sam3 package must be installed — it already is in this repo)
"""

import sys
import os
import time
import argparse
import glob
from collections import OrderedDict
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image

# ── SAM3 imports ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from importlib.resources import files as importlib_files

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


# ─────────────────────────────────────────────────────────────────────────────
# Centroid tracker  (identical to crystal_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=90, max_distance=200):
        self.next_id       = 0
        self.objects       = OrderedDict()
        self.disappeared   = OrderedDict()
        self.graveyard     = []
        self.graveyard_ttl = 300
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.total_seen      = 0

    def register(self, centroid):
        best_dist, best_idx = float('inf'), -1
        for i, (gc, _) in enumerate(self.graveyard):
            d = np.hypot(centroid[0] - gc[0], centroid[1] - gc[1])
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_dist < self.max_distance * 1.5 and best_idx >= 0:
            self.graveyard.pop(best_idx)
        else:
            self.total_seen += 1
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        self.graveyard.append((self.objects[oid], 0))
        del self.objects[oid]
        del self.disappeared[oid]

    def _age_graveyard(self):
        self.graveyard = [(c, a + 1) for c, a in self.graveyard if a < self.graveyard_ttl]

    def update(self, centroids):
        self._age_graveyard()
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        if not self.objects:
            for c in centroids:
                self.register(c)
            return self.objects
        obj_ids = list(self.objects.keys())
        A = np.array(list(self.objects.values()), dtype=float)
        B = np.array(centroids, dtype=float)
        D = np.sqrt(((A[:, None] - B[None, :]) ** 2).sum(axis=2))
        rows  = D.min(axis=1).argsort()
        cols  = D.argmin(axis=1)[rows]
        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = obj_ids[r]
            self.objects[oid]     = centroids[c]
            self.disappeared[oid] = 0
            used_r.add(r); used_c.add(c)
        for r in set(range(D.shape[0])) - used_r:
            self.disappeared[obj_ids[r]] += 1
            if self.disappeared[obj_ids[r]] > self.max_disappeared:
                self.deregister(obj_ids[r])
        for c in set(range(D.shape[1])) - used_c:
            self.register(centroids[c])
        return self.objects

    def reset(self):
        self.objects.clear(); self.disappeared.clear()
        self.graveyard.clear()
        self.next_id = 0; self.total_seen = 0


# ─────────────────────────────────────────────────────────────────────────────
# Slider panel  (adapted from crystal_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
class SliderPanel:
    PANEL_H  = 460
    PAD_L    = 200
    PAD_R    = 60
    ROW_H    = 36
    TRACK_H  = 6
    HANDLE_R = 10

    SCRUB_H  = 52
    SCRUB_PAD = 10

    def __init__(self, width, params, total_frames=0, fps=30):
        self.width  = width
        self.params = params
        self.dragging      = None
        self.total_frames  = max(1, total_frames)
        self.fps           = max(1, fps)
        self.current_frame = 0
        self.seek_frame    = None

    def _track_x(self, idx):
        return self.PAD_L, self.width - self.PAD_R

    def _val_to_x(self, idx):
        p  = self.params[idx]
        x0, x1 = self._track_x(idx)
        t  = (p['value'] - p['min']) / max(1, p['max'] - p['min'])
        return int(x0 + t * (x1 - x0))

    def _x_to_val(self, idx, x):
        p  = self.params[idx]
        x0, x1 = self._track_x(idx)
        t  = np.clip((x - x0) / max(1, x1 - x0), 0, 1)
        raw = p['min'] + t * (p['max'] - p['min'])
        # respect step if present
        step = p.get('step', 1)
        if step >= 1:
            return int(round(raw / step) * step)
        return float(f"{raw:.3f}")

    def _row_y(self, idx):
        return self.SCRUB_H + 30 + idx * self.ROW_H * 2

    def _scrub_x(self):
        pad = self.SCRUB_PAD
        x0, x1 = pad, self.width - pad
        t = self.current_frame / self.total_frames
        return int(x0 + t * (x1 - x0)), x0, x1

    def _x_to_frame(self, x):
        pad = self.SCRUB_PAD
        x0, x1 = pad, self.width - pad
        t = np.clip((x - x0) / max(1, x1 - x0), 0, 1)
        return int(t * self.total_frames)

    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            hx, sx0, sx1 = self._scrub_x()
            sy = self.SCRUB_H // 2
            if sx0 <= x <= sx1 and abs(y - sy) < 20:
                self.dragging = 'scrub'
                self.seek_frame = self._x_to_frame(x)
                return
            for i in range(len(self.params)):
                ry = self._row_y(i)
                hx2 = self._val_to_x(i)
                x0, x1 = self._track_x(i)
                if abs(x - hx2) < self.HANDLE_R + 8 and abs(y - ry) < self.HANDLE_R + 8:
                    self.dragging = i; break
                if x0 <= x <= x1 and abs(y - ry) < 16:
                    self.params[i]['value'] = self._x_to_val(i, x)
                    self.dragging = i; break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging == 'scrub':
                self.seek_frame = self._x_to_frame(x)
            elif self.dragging is not None:
                self.params[self.dragging]['value'] = self._x_to_val(self.dragging, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    def _fmt_time(self, frame):
        secs = int(frame / self.fps)
        return f"{secs // 60}:{secs % 60:02d}"

    def render(self):
        canvas = np.zeros((self.PANEL_H, self.width, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 35)

        # Scrubber
        cv2.rectangle(canvas, (0, 0), (self.width, self.SCRUB_H), (18, 18, 28), -1)
        cv2.putText(canvas, "VIDEO POSITION", (10, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1)
        pad = self.SCRUB_PAD
        sx0, sx1 = pad, self.width - pad
        sy = self.SCRUB_H // 2 + 5
        cv2.line(canvas, (sx0, sy), (sx1, sy), (55, 55, 70), 4)
        hx, _, _ = self._scrub_x()
        cv2.line(canvas, (sx0, sy), (hx, sy), (80, 140, 255), 4)
        col = (140, 200, 255) if self.dragging == 'scrub' else (100, 170, 255)
        cv2.circle(canvas, (hx, sy), 10, col, -1)
        cv2.circle(canvas, (hx, sy), 10, (255, 255, 255), 1)
        cv2.putText(canvas, self._fmt_time(self.current_frame), (sx0, sy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 200, 255), 1)
        cv2.putText(canvas, self._fmt_time(self.total_frames), (sx1 - 35, sy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 130), 1)
        cv2.line(canvas, (0, self.SCRUB_H), (self.width, self.SCRUB_H), (45, 45, 60), 1)

        # Param header
        cv2.rectangle(canvas, (0, self.SCRUB_H), (self.width, self.SCRUB_H + 22), (40, 40, 55), -1)
        cv2.putText(canvas, "PARAMETERS  —  drag sliders to adjust",
                    (10, self.SCRUB_H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1)

        for i, p in enumerate(self.params):
            ry = self._row_y(i)
            x0, x1 = self._track_x(i)
            hx2 = self._val_to_x(i)
            cv2.putText(canvas, p['desc'], (x0, ry - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 140), 1)
            cv2.putText(canvas, p['name'], (8, ry + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 230), 1)
            cv2.line(canvas, (x0, ry), (x1, ry), (60, 60, 75), self.TRACK_H)
            cv2.line(canvas, (x0, ry), (hx2, ry), (0, 180, 120), self.TRACK_H)
            col = (0, 255, 180) if self.dragging == i else (0, 210, 140)
            cv2.circle(canvas, (hx2, ry), self.HANDLE_R, col, -1)
            cv2.circle(canvas, (hx2, ry), self.HANDLE_R, (255, 255, 255), 1)
            val = p['value']
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            cv2.putText(canvas, val_str, (x1 + 8, ry + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 1)

        cv2.putText(canvas,
                    "q=quit   r=reset box   s=save frame   SPACE=pause/resume",
                    (10, self.PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 120), 1)
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Box drawer  (for reference crystal ROI)
# ─────────────────────────────────────────────────────────────────────────────
class BoxDrawer:
    """Lets the user rubber-band a rectangle on the video frame."""

    def __init__(self):
        self.pt1      = None
        self.pt2      = None
        self.drawing  = False
        self.confirmed = False

    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt1 = (x, y); self.pt2 = (x, y)
            self.drawing = True; self.confirmed = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.pt2 = (x, y)
            self.drawing = False; self.confirmed = True

    def reset(self):
        self.pt1 = self.pt2 = None
        self.drawing = self.confirmed = False

    @property
    def box(self):
        """(x, y, w, h) in pixel coords, or None if not set."""
        if self.pt1 is None or self.pt2 is None:
            return None
        x1 = min(self.pt1[0], self.pt2[0])
        y1 = min(self.pt1[1], self.pt2[1])
        x2 = max(self.pt1[0], self.pt2[0])
        y2 = max(self.pt1[1], self.pt2[1])
        w, h = x2 - x1, y2 - y1
        return (x1, y1, w, h) if w > 5 and h > 5 else None

    def draw_on(self, img):
        box = self.box
        if box is None:
            return
        x, y, w, h = box
        color = (0, 255, 255) if self.drawing else (255, 200, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = "drawing..." if self.drawing else "REFERENCE  (ENTER to confirm)"
        cv2.putText(img, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
MASK_COLORS = [
    (0, 200, 100), (255, 120, 0), (0, 160, 255),
    (220, 0, 220), (0, 220, 220), (220, 200, 0),
]

def overlay_masks(display, masks_np, boxes_np, scores_np, tracked, scale_xy):
    """Draw SAM3 masks + bounding boxes + tracker IDs onto display."""
    sx, sy = scale_xy
    n = len(masks_np)
    for i in range(n):
        mask = masks_np[i]           # (H, W) bool
        score = float(scores_np[i])
        bx0, by0, bx1, by1 = (int(v) for v in boxes_np[i])
        # Scale boxes to display resolution
        bx0d = int(bx0 * sx); by0d = int(by0 * sy)
        bx1d = int(bx1 * sx); by1d = int(by1 * sy)
        cx = (bx0d + bx1d) // 2
        cy = (by0d + by1d) // 2

        col = MASK_COLORS[i % len(MASK_COLORS)]

        # Semi-transparent mask
        mask_small = cv2.resize(
            mask.astype(np.uint8),
            (display.shape[1], display.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        overlay = display.copy()
        overlay[mask_small] = (
            overlay[mask_small].astype(float) * 0.45 +
            np.array(col, dtype=float) * 0.55
        ).astype(np.uint8)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Bounding box
        cv2.rectangle(display, (bx0d, by0d), (bx1d, by1d), col, 2)
        cv2.putText(display, f"{score:.2f}", (bx0d + 4, by0d + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # Tracker IDs
    for oid, (cx, cy) in tracked.items():
        cxd, cyd = int(cx * sx), int(cy * sy)
        cv2.circle(display, (cxd, cyd), 5, (0, 0, 255), -1)
        cv2.putText(display, f"#{oid}", (cxd + 7, cyd - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 0), 1)


def draw_hud(display, n_frame, total_seen, fps, paused, confidence):
    hud = display.copy()
    cv2.rectangle(hud, (6, 6), (340, 140), (0, 0, 0), -1)
    cv2.addWeighted(hud, 0.45, display, 0.55, 0, display)
    cv2.putText(display, "CRYSTAL DETECTOR — SAM3", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    ptag = "  [PAUSED]" if paused else ""
    cv2.putText(display, f"In frame:   {n_frame:3d}{ptag}", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    cv2.putText(display, f"Total seen: {total_seen:3d}", (14, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)
    cv2.putText(display, f"FPS: {fps:.1f}   conf≥{confidence:.2f}", (14, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(display, "r=new ref  s=save  SPACE=pause  q=quit", (14, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 140), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_arg, device):
    bpe_path = str(importlib_files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

    if checkpoint_arg and os.path.exists(checkpoint_arg):
        ckpt = checkpoint_arg
    else:
        # Try local sam3.pt in project root
        local = os.path.join(PROJECT_ROOT, "sam3.pt")
        if os.path.exists(local):
            ckpt = local
        else:
            ckpt = None

    if ckpt:
        print(f"Loading SAM3 from local checkpoint: {ckpt}")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=ckpt,
            load_from_HF=False,
            device=str(device),
            eval_mode=True,
        )
    else:
        print("sam3.pt not found locally — downloading from HuggingFace...")
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            load_from_HF=True,
            device=str(device),
            eval_mode=True,
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Setup screen — let user draw reference box
# ─────────────────────────────────────────────────────────────────────────────
def run_setup_screen(frame_bgr, ref_image_override=None):
    """
    Shows the frame (or ref_image_override) and lets the user draw a box.
    Returns (pil_image, box_xywh) or (None, None) if user quits.
    """
    if ref_image_override is not None:
        base = cv2.imread(ref_image_override)
        if base is None:
            print(f"Could not read reference image: {ref_image_override}")
            base = frame_bgr.copy()
    else:
        base = frame_bgr.copy()

    h, w = base.shape[:2]
    MAX_W = 1200
    scale = min(1.0, MAX_W / w)
    dw, dh = int(w * scale), int(h * scale)
    small = cv2.resize(base, (dw, dh))

    drawer = BoxDrawer()
    WIN_SETUP = "Crystal Detector — Draw box around reference crystal, then press ENTER"
    cv2.namedWindow(WIN_SETUP, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_SETUP, dw, dh)

    def cb(event, x, y, flags, param):
        drawer.on_mouse(event, x, y)

    cv2.setMouseCallback(WIN_SETUP, cb)

    while True:
        canvas = small.copy()
        drawer.draw_on(canvas)

        # Instructions overlay
        inst_lines = [
            "Draw a box around ONE reference crystal",
            "ENTER / SPACE = confirm   r = redraw   q = quit",
        ]
        for li, line in enumerate(inst_lines):
            cv2.putText(canvas, line, (12, 24 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(canvas, line, (12, 24 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(WIN_SETUP, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            cv2.destroyWindow(WIN_SETUP)
            return None, None

        if key == ord('r'):
            drawer.reset()

        if key in (13, ord(' ')):          # ENTER or SPACE
            box = drawer.box
            if box is not None:
                # Scale box back to original image coordinates
                inv = 1.0 / scale
                x, y, bw, bh = (int(v * inv) for v in box)
                cv2.destroyWindow(WIN_SETUP)
                pil_img = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
                return pil_img, (x, y, bw, bh)
            else:
                # Flash a warning
                cv2.putText(canvas, "  Draw a box first!", (dw // 2 - 120, dh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 50, 255), 2)
                cv2.imshow(WIN_SETUP, canvas)
                cv2.waitKey(600)


# ─────────────────────────────────────────────────────────────────────────────
# Mouse relay
# ─────────────────────────────────────────────────────────────────────────────
class MouseRelay:
    def __init__(self, panel, video_h):
        self.panel   = panel
        self.video_h = video_h

    def callback(self, event, x, y, flags, param):
        panel_y = y - self.video_h
        if panel_y >= 0:
            self.panel.on_mouse(event, x, panel_y)


# ─────────────────────────────────────────────────────────────────────────────
# Camera finder
# ─────────────────────────────────────────────────────────────────────────────
def find_best_camera():
    best_idx, best_px = 0, 0
    found = []
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, ww, hh))
                if ww * hh > best_px:
                    best_px = ww * hh; best_idx = i
        cap.release()
    if not found:
        print("ERROR: No cameras found.")
        return None
    for i, ww, hh in found:
        tag = " <- selected" if i == best_idx else ""
        print(f"  Camera {i}: {ww}x{hh}{tag}")
    return best_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(source, camera_index=0, ref_image=None, checkpoint=None):
    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else nullcontext()
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(checkpoint, device)

    # ── Open video source ─────────────────────────────────────────────────────
    if source == 'camera':
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vid_h, vid_w = first_frame.shape[:2]
    MAX_W = 1200
    scale     = min(1.0, MAX_W / vid_w)
    disp_w    = int(vid_w * scale)
    disp_h    = int(vid_h * scale)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    # ── Sliders ───────────────────────────────────────────────────────────────
    params = [
        {'name': 'Confidence',     'min': 5,   'max': 95,  'value': 50,  'step': 1,
         'desc': 'SAM3 score threshold (x0.01). Lower = more detections, may add noise.'},
        {'name': 'Min mask area',  'min': 50,  'max': 5000,'value': 200, 'step': 50,
         'desc': 'Ignore detections smaller than this many pixels.'},
        {'name': 'Process every',  'min': 1,   'max': 30,  'value': 3,   'step': 1,
         'desc': 'Run SAM3 every N frames (higher = faster but less smooth).'},
        {'name': 'Memory (frames)','min': 5,   'max': 300, 'value': 90,  'step': 5,
         'desc': 'How long to remember a crystal after it disappears.'},
        {'name': 'Match radius',   'min': 10,  'max': 500, 'value': 200, 'step': 10,
         'desc': 'Max pixels a crystal can move between frames and still match.'},
    ]

    panel    = SliderPanel(disp_w, params, total_frames=total_frames, fps=video_fps)
    total_h  = disp_h + SliderPanel.PANEL_H
    WIN = "Crystal Detector  |  SAM3"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, total_h)

    tracker   = CentroidTracker(max_disappeared=90, max_distance=200)
    relay     = MouseRelay(panel, disp_h)
    cv2.setMouseCallback(WIN, relay.callback)

    # ── State ─────────────────────────────────────────────────────────────────
    saved_prompt      = None
    saved_prompt_mask = None
    processor         = None
    paused      = False
    last_frame  = first_frame.copy()
    last_masks  = []
    last_boxes  = []
    last_scores = []
    last_tracked = {}
    frame_count  = 0
    save_count   = 0
    fps_val      = 0.0
    fps_timer    = time.time()

    need_ref = True   # True = show setup screen before next frame

    print("Crystal SAM3 Detector started.")
    print("Draw a reference box on the first frame.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    with autocast_ctx:
        while True:
            # -------- Reference setup ----------------------------------------
            if need_ref:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, ref_frame = cap.read()
                if not ret:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ref_pil, box_xywh = run_setup_screen(ref_frame, ref_image_override=ref_image)
                if ref_pil is None:
                    break  # user quit

                print(f"Reference box: {box_xywh}")
                print("Extracting SAM3 reference prompt...")

                confidence = params[0]['value'] / 100.0
                processor  = Sam3Processor(model, confidence_threshold=confidence)

                state1 = processor.set_image(ref_pil)
                w1, h1 = ref_pil.size
                x, y, bw, bh = box_xywh
                box_t    = torch.tensor([x, y, bw, bh], dtype=torch.float32).view(-1, 4)
                box_cx   = box_xywh_to_cxcywh(box_t)
                norm_box = normalize_bbox(box_cx, w1, h1).flatten().tolist()
                state1   = processor._add_box_prompt(box=norm_box, label=True, state=state1)
                state1   = processor._forward_grounding(state1.copy())

                saved_prompt      = state1["prompt"]
                saved_prompt_mask = state1["prompt_mask"]

                tracker.reset()
                last_masks  = []; last_boxes = []; last_scores = []; last_tracked = {}
                frame_count = 0
                paused      = False
                need_ref    = False
                print("Reference set. Detecting crystals...")

            # -------- Handle scrubber seek -----------------------------------
            if panel.seek_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, panel.seek_frame)
                panel.current_frame = panel.seek_frame
                panel.seek_frame    = None
                paused = True

            # -------- Read frame ---------------------------------------------
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if source != 'camera':
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        panel.current_frame = 0
                        continue
                    break
                last_frame  = frame.copy()
                frame_count += 1
                panel.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                frame = last_frame.copy()

            # -------- Read slider values -------------------------------------
            confidence = params[0]['value'] / 100.0
            min_area   = params[1]['value']
            every_n    = params[2]['value']
            memory     = params[3]['value']
            match_r    = params[4]['value']

            tracker.max_disappeared = memory
            tracker.max_distance    = match_r

            # -------- SAM3 inference (every N frames) -----------------------
            run_sam = (not paused) and (frame_count % every_n == 0)
            if run_sam and saved_prompt is not None:
                processor.confidence_threshold = confidence
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                state2 = processor.set_image(frame_pil)
                state2["prompt"]      = saved_prompt
                state2["prompt_mask"] = saved_prompt_mask
                state2 = processor._forward_grounding(state2)

                masks_t  = state2.get("masks")   # (N,1,H,W) bool tensor or None
                boxes_t  = state2.get("boxes")   # (N,4) tensor
                scores_t = state2.get("scores")  # (N,) tensor

                if masks_t is not None and len(masks_t) > 0:
                    masks_np  = masks_t.squeeze(1).cpu().numpy().astype(bool)
                    boxes_np  = boxes_t.cpu().numpy()
                    scores_np = scores_t.cpu().numpy()

                    # Filter by min mask area
                    keep = []
                    for i, m in enumerate(masks_np):
                        if int(m.sum()) >= min_area:
                            keep.append(i)
                    masks_np  = masks_np[keep]
                    boxes_np  = boxes_np[keep]
                    scores_np = scores_np[keep]
                else:
                    masks_np = np.zeros((0,), dtype=bool)
                    boxes_np  = np.zeros((0, 4))
                    scores_np = np.zeros((0,))

                last_masks  = masks_np
                last_boxes  = boxes_np
                last_scores = scores_np

                # Centroids from box centres
                centroids = []
                for bx0, by0, bx1, by1 in boxes_np:
                    centroids.append((float((bx0 + bx1) / 2), float((by0 + by1) / 2)))
                last_tracked = dict(tracker.update(centroids))

                # FPS
                if frame_count % 10 == 0:
                    now = time.time()
                    fps_val   = 10 / max(1e-3, now - fps_timer)
                    fps_timer = now

            # -------- Render -------------------------------------------------
            display = cv2.resize(frame, (disp_w, disp_h))
            sx, sy  = disp_w / vid_w, disp_h / vid_h

            if len(last_masks) > 0:
                overlay_masks(display, last_masks, last_boxes, last_scores,
                              last_tracked, (sx, sy))

            draw_hud(display, len(last_masks), tracker.total_seen, fps_val, paused, confidence)

            slider_canvas = panel.render()
            combined = np.vstack([display, slider_canvas])
            cv2.imshow(WIN, combined)

            # -------- Key handling -------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                need_ref = True
                print("Returning to reference selection...")
            elif key == ord('s'):
                fname = f"crystal_sam_{save_count:04d}.png"
                cv2.imwrite(fname, combined)
                save_count += 1
                print(f"Saved {fname}")
            elif key == ord(' '):
                paused = not paused
                print("Paused." if paused else "Resumed.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete — {tracker.total_seen} unique crystals tracked.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time crystal detector — SAM3")
    parser.add_argument("--camera",     type=int,   default=None,  help="Camera index")
    parser.add_argument("--video",      type=str,   default=None,  help="Path to video file")
    parser.add_argument("--ref",        type=str,   default=None,  help="Separate reference image (optional)")
    parser.add_argument("--checkpoint", type=str,   default=None,  help="Path to sam3.pt checkpoint")
    args = parser.parse_args()

    if args.video:
        run(source=args.video, ref_image=args.ref, checkpoint=args.checkpoint)
    else:
        idx = args.camera if args.camera is not None else find_best_camera()
        if idx is not None:
            run(source='camera', camera_index=idx, ref_image=args.ref, checkpoint=args.checkpoint)
