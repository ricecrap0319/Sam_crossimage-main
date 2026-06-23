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
import threading
from collections import OrderedDict
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image
import mss

import tkinter as tk
from tkinter import filedialog

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
    def __init__(self, max_disappeared=120, max_distance=300):
        self.next_id     = 0
        self.objects     = OrderedDict()
        self.velocity    = OrderedDict()
        self.disappeared = OrderedDict()
        self.graveyard   = []
        self.graveyard_ttl   = 400
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.total_seen      = 0

    def _predicted(self, oid):
        cx, cy = self.objects[oid]
        vx, vy = self.velocity.get(oid, (0, 0))
        age = self.disappeared[oid]
        return cx + vx * (age + 1), cy + vy * (age + 1)

    def register(self, centroid, vel=(0, 0)):
        best_dist, best_idx = float('inf'), -1
        for i, (gc, gv, _) in enumerate(self.graveyard):
            d = np.hypot(centroid[0]-gc[0], centroid[1]-gc[1])
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_dist < self.max_distance * 1.5 and best_idx >= 0:
            _, gv, _ = self.graveyard.pop(best_idx)
            vel = gv
        else:
            self.total_seen += 1
        self.objects[self.next_id]     = centroid
        self.velocity[self.next_id]    = vel
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        self.graveyard.append((self.objects[oid], self.velocity[oid], 0))
        del self.objects[oid]; del self.velocity[oid]; del self.disappeared[oid]

    def _age_graveyard(self):
        self.graveyard = [(c, v, a+1) for c, v, a in self.graveyard if a < self.graveyard_ttl]

    def update(self, centroids):
        self._age_graveyard()
        if not centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        if not self.objects:
            for c in centroids: self.register(c)
            return self.objects
        obj_ids = list(self.objects.keys())
        A = np.array([self._predicted(oid) for oid in obj_ids], dtype=float)
        B = np.array(centroids, dtype=float)
        D = np.sqrt(((A[:,None]-B[None,:])**2).sum(2))
        rows = D.min(1).argsort(); cols = D.argmin(1)[rows]
        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c: continue
            if D[r, c] > self.max_distance: continue
            oid = obj_ids[r]
            old_cx, old_cy = self.objects[oid]
            new_cx, new_cy = centroids[c]
            old_vx, old_vy = self.velocity[oid]
            alpha = 0.5
            self.velocity[oid]    = (alpha*(new_cx-old_cx)+(1-alpha)*old_vx,
                                     alpha*(new_cy-old_cy)+(1-alpha)*old_vy)
            self.objects[oid]     = centroids[c]
            self.disappeared[oid] = 0
            used_r.add(r); used_c.add(c)
        for r in set(range(D.shape[0]))-used_r:
            oid = obj_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)
        for c in set(range(D.shape[1]))-used_c:
            self.register(centroids[c])
        return self.objects

    def reset(self):
        self.objects.clear(); self.velocity.clear()
        self.disappeared.clear(); self.graveyard.clear()
        self.next_id = 0; self.total_seen = 0


# ─────────────────────────────────────────────────────────────────────────────
# Slider panel  (adapted from crystal_detector.py)
# ─────────────────────────────────────────────────────────────────────────────
class SliderPanel:
    PANEL_H   = 460
    PAD_L     = 210
    PAD_R     = 200
    ROW_H     = 36
    TRACK_H   = 6
    HANDLE_R  = 10
    SCRUB_H   = 52
    SCRUB_PAD = 10
    BTN_W     = 26
    BTN_H     = 22

    def __init__(self, width, params, total_frames=0, fps=30):
        self.width         = width
        self.params        = params
        self.dragging      = None
        self.edit_idx      = None
        self.edit_buf      = ""
        self.total_frames  = max(1, total_frames)
        self.fps           = max(1, fps)
        self.current_frame = 0
        self.seek_frame    = None

    def _tx(self, idx):
        return self.PAD_L, self.width - self.PAD_R

    def _val_to_x(self, idx):
        p = self.params[idx]; x0, x1 = self._tx(idx)
        return int(x0 + (p['value']-p['min']) / max(1, p['max']-p['min']) * (x1-x0))

    def _x_to_val(self, idx, x):
        p = self.params[idx]; x0, x1 = self._tx(idx)
        t   = np.clip((x-x0) / max(1, x1-x0), 0, 1)
        raw = p['min'] + t * (p['max']-p['min'])
        s   = p.get('step', 1)
        return int(round(raw/s)*s) if s >= 1 else float(f"{raw:.3f}")

    def _row_y(self, idx):
        return self.SCRUB_H + 30 + idx * self.ROW_H * 2

    def _btn_minus_rect(self, idx):
        x1_track = self.width - self.PAD_R
        bx = x1_track + 8; ry = self._row_y(idx)
        return bx, ry-self.BTN_H//2, bx+self.BTN_W, ry+self.BTN_H//2

    def _btn_plus_rect(self, idx):
        bx = self.width - self.PAD_R + 8 + self.BTN_W + 52 + 8; ry = self._row_y(idx)
        return bx, ry-self.BTN_H//2, bx+self.BTN_W, ry+self.BTN_H//2

    def _val_box_rect(self, idx):
        bx0 = self.width - self.PAD_R + 8 + self.BTN_W + 4; bx1 = bx0 + 52
        ry  = self._row_y(idx)
        return bx0, ry-self.BTN_H//2, bx1, ry+self.BTN_H//2

    def _clamp(self, idx, val):
        p = self.params[idx]; s = p.get('step', 1)
        val = max(p['min'], min(p['max'], val))
        return int(round(val/s)*s) if s >= 1 else float(f"{val:.3f}")

    def _scrub_x(self):
        pad = self.SCRUB_PAD; x0, x1 = pad, self.width-pad
        return int(x0 + self.current_frame/self.total_frames*(x1-x0)), x0, x1

    def _x_to_frame(self, x):
        pad = self.SCRUB_PAD; x0, x1 = pad, self.width-pad
        return int(np.clip((x-x0)/max(1,x1-x0),0,1)*self.total_frames)

    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._commit_edit()
            hx, sx0, sx1 = self._scrub_x()
            if sx0<=x<=sx1 and abs(y-self.SCRUB_H//2)<20:
                self.dragging='scrub'; self.seek_frame=self._x_to_frame(x); return
            for i in range(len(self.params)):
                ry = self._row_y(i)
                mx0,my0,mx1,my1 = self._btn_minus_rect(i)
                if mx0<=x<=mx1 and my0<=y<=my1:
                    self.params[i]['value']=self._clamp(i,self.params[i]['value']-self.params[i].get('step',1)); return
                px0,py0,px1,py1 = self._btn_plus_rect(i)
                if px0<=x<=px1 and py0<=y<=py1:
                    self.params[i]['value']=self._clamp(i,self.params[i]['value']+self.params[i].get('step',1)); return
                vx0,vy0,vx1,vy1 = self._val_box_rect(i)
                if vx0<=x<=vx1 and vy0<=y<=vy1:
                    self.edit_idx=i; self.edit_buf=str(self.params[i]['value']); return
                hx2=self._val_to_x(i); x0t,x1t=self._tx(i)
                if abs(x-hx2)<self.HANDLE_R+8 and abs(y-ry)<self.HANDLE_R+8:
                    self.dragging=i; return
                if x0t<=x<=x1t and abs(y-ry)<16:
                    self.params[i]['value']=self._x_to_val(i,x); self.dragging=i; return
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging=='scrub': self.seek_frame=self._x_to_frame(x)
            elif self.dragging is not None: self.params[self.dragging]['value']=self._x_to_val(self.dragging,x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging=None

    def on_key(self, key):
        if self.edit_idx is None: return False
        if key in (13,10):  self._commit_edit(); return True
        if key==27:         self.edit_idx=None; self.edit_buf=""; return True
        if key==8:          self.edit_buf=self.edit_buf[:-1]; return True
        if 48<=key<=57:     self.edit_buf+=chr(key); return True
        if key==ord('.') and '.' not in self.edit_buf: self.edit_buf+='.'; return True
        if key==ord('-') and self.edit_buf=="": self.edit_buf='-'; return True
        return True

    def _commit_edit(self):
        if self.edit_idx is None: return
        try:
            val=float(self.edit_buf)
            self.params[self.edit_idx]['value']=self._clamp(self.edit_idx,val)
        except ValueError: pass
        self.edit_idx=None; self.edit_buf=""

    def _fmt_time(self, frame):
        secs=int(frame/self.fps); return f"{secs//60}:{secs%60:02d}"

    def render(self, show_scrubber=True):
        c=np.zeros((self.PANEL_H,self.width,3),dtype=np.uint8); c[:]=(25,25,35)
        # Scrubber
        cv2.rectangle(c,(0,0),(self.width,self.SCRUB_H),(18,18,28),-1)
        if show_scrubber:
            cv2.putText(c,"VIDEO POSITION",(10,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,120,150),1)
            pad=self.SCRUB_PAD; sx0,sx1=pad,self.width-pad; sy=self.SCRUB_H//2+5
            cv2.line(c,(sx0,sy),(sx1,sy),(55,55,70),4)
            hx,_,_=self._scrub_x(); cv2.line(c,(sx0,sy),(hx,sy),(80,140,255),4)
            col=(140,200,255) if self.dragging=='scrub' else (100,170,255)
            cv2.circle(c,(hx,sy),10,col,-1); cv2.circle(c,(hx,sy),10,(255,255,255),1)
            cv2.putText(c,self._fmt_time(self.current_frame),(sx0,sy-14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,200,255),1)
            cv2.putText(c,self._fmt_time(self.total_frames),(sx1-35,sy-14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,130),1)
        else:
            cv2.putText(c,"LIVE MODE",(10,self.SCRUB_H//2+6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,150),1)
        cv2.line(c,(0,self.SCRUB_H),(self.width,self.SCRUB_H),(45,45,60),1)
        cv2.rectangle(c,(0,self.SCRUB_H),(self.width,self.SCRUB_H+22),(40,40,55),-1)
        cv2.putText(c,"PARAMETERS — drag slider  |  click value to type  |  use  −  +  buttons",
                    (10,self.SCRUB_H+16),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,200),1)
        for i,p in enumerate(self.params):
            ry=self._row_y(i); x0t,x1t=self._tx(i); hx2=self._val_to_x(i)
            editing=(self.edit_idx==i)
            cv2.putText(c,p['desc'],(x0t,ry-12),cv2.FONT_HERSHEY_SIMPLEX,0.31,(120,120,140),1)
            cv2.putText(c,p['name'],(8,ry+4),cv2.FONT_HERSHEY_SIMPLEX,0.44,(210,210,230),1)
            cv2.line(c,(x0t,ry),(x1t,ry),(60,60,75),self.TRACK_H)
            cv2.line(c,(x0t,ry),(hx2,ry),(0,180,120),self.TRACK_H)
            hcol=(0,255,180) if self.dragging==i else (0,210,140)
            cv2.circle(c,(hx2,ry),self.HANDLE_R,hcol,-1); cv2.circle(c,(hx2,ry),self.HANDLE_R,(255,255,255),1)
            mx0,my0,mx1,my1=self._btn_minus_rect(i)
            cv2.rectangle(c,(mx0,my0),(mx1,my1),(60,60,80),-1); cv2.rectangle(c,(mx0,my0),(mx1,my1),(140,140,160),1)
            cv2.putText(c,"-",(mx0+6,my1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,100,100),1)
            vx0,vy0,vx1,vy1=self._val_box_rect(i)
            bg=(50,50,80) if editing else (35,35,55); border=(0,200,255) if editing else (100,100,130)
            cv2.rectangle(c,(vx0,vy0),(vx1,vy1),bg,-1); cv2.rectangle(c,(vx0,vy0),(vx1,vy1),border,1)
            disp=(self.edit_buf+"|") if editing else (f"{p['value']:.1f}" if isinstance(p['value'],float) else str(p['value']))
            cv2.putText(c,disp,(vx0+4,vy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,180),1)
            px0,py0,px1,py1=self._btn_plus_rect(i)
            cv2.rectangle(c,(px0,py0),(px1,py1),(60,60,80),-1); cv2.rectangle(c,(px0,py0),(px1,py1),(140,140,160),1)
            cv2.putText(c,"+",(px0+5,py1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,220,100),1)
        cv2.putText(c,"q=quit  r=new ref  s=save  SPACE=pause/resume  (click value box to type, ENTER to confirm)",
                    (10,self.PANEL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,100,120),1)
        return c


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
# File picker (for video files)
# ─────────────────────────────────────────────────────────────────────────────
def pick_video(title="Select video file"):
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


# ─────────────────────────────────────────────────────────────────────────────
# Source selection screen
# ─────────────────────────────────────────────────────────────────────────────
def source_select_screen():
    """Returns ('camera', idx) | ('video', path) | ('screen', None) | None."""
    WIN = "Crystal Detector — Select Source"
    canvas = np.zeros((360, 680, 3), dtype=np.uint8); canvas[:] = (25, 25, 35)
    items = [
        ("CRYSTAL DETECTOR — SAM3",            (255, 255, 255), 0.75, 2),
        ("",                                    None,            0,    0),
        ("L  —  Live camera",                   (0, 255, 180),   0.62, 1),
        ("V  —  Video file  (file picker)",     (0, 200, 255),   0.62, 1),
        ("S  —  Screen capture  (NIS-Elements)",(200, 180, 255), 0.62, 1),
        ("Q  —  Quit",                          (180, 80,  80),  0.5,  1),
    ]
    y = 55
    for text, col, sc, th in items:
        if col: cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)
        y += 50
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 680, 360)
    cv2.imshow(WIN, canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):   cv2.destroyWindow(WIN); return None
        if key == ord('l'):         cv2.destroyWindow(WIN); return ('camera', None)
        if key == ord('v'):
            cv2.destroyWindow(WIN)
            path = pick_video("Select video file")
            if path and os.path.exists(path): return ('video', path)
            print("No video selected."); return None
        if key == ord('s'):         cv2.destroyWindow(WIN); return ('screen', None)


# ─────────────────────────────────────────────────────────────────────────────
# Screen region selector
# ─────────────────────────────────────────────────────────────────────────────
def select_screen_region():
    with mss.MSS() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        full_bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    scale = min(1.0, 1600 / full_bgr.shape[1])
    small = cv2.resize(full_bgr, (int(full_bgr.shape[1]*scale), int(full_bgr.shape[0]*scale)))
    state = {'pt1': None, 'pt2': None, 'drawing': False, 'done': False}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['pt1']=(x,y); state['pt2']=(x,y); state['drawing']=True
        elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
            state['pt2']=(x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            state['pt2']=(x,y); state['drawing']=False; state['done']=True

    WIN = "Select capture region — drag over the microscope live view, then ENTER"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, small.shape[1], small.shape[0])
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        canvas = small.copy()
        cv2.putText(canvas, "Drag a box over the microscope live view, then press ENTER",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(canvas, "Drag a box over the microscope live view, then press ENTER",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 1)
        if state['pt1'] and state['pt2']:
            col = (0,255,255) if state['drawing'] else (0,200,255)
            cv2.rectangle(canvas, state['pt1'], state['pt2'], col, 2)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27): cv2.destroyWindow(WIN); return None
        if key in (13, ord(' ')) and state['done'] and state['pt1'] and state['pt2']:
            x1=min(state['pt1'][0],state['pt2'][0]); y1=min(state['pt1'][1],state['pt2'][1])
            x2=max(state['pt1'][0],state['pt2'][0]); y2=max(state['pt1'][1],state['pt2'][1])
            if (x2-x1)<10 or (y2-y1)<10: continue
            inv = 1.0/scale
            region = {'left': int(x1*inv)+monitor['left'], 'top': int(y1*inv)+monitor['top'],
                      'width': int((x2-x1)*inv), 'height': int((y2-y1)*inv)}
            cv2.destroyWindow(WIN)
            print(f"Capture region: {region}")
            return region


# ─────────────────────────────────────────────────────────────────────────────
# Screen capture loop (same layout as main loop but reads from mss)
# ─────────────────────────────────────────────────────────────────────────────
def run_screen_loop(model, autocast_ctx, ref_image=None):
    """Screen capture detection loop — identical UI to the camera/video loop."""

    # ── Step 1: pick reference image from file ────────────────────────────────
    if ref_image and os.path.exists(ref_image):
        ref_path = ref_image
    else:
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        ref_path = filedialog.askopenfilename(
            title="Select reference image (photo of a crystal)",
            initialdir=os.path.expanduser("~"),
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All files", "*.*")]
        )
        root.destroy()
        if not ref_path:
            print("No reference image selected."); return

    # ── Step 2: draw box on reference image ───────────────────────────────────
    ref_pil, box_xywh = run_setup_screen(cv2.imread(ref_path), ref_image_override=ref_path)
    if ref_pil is None: return

    # ── Step 3: show loading screen while extracting prompt ───────────────────
    WIN_LOAD = "Crystal Detector — Loading..."
    loading = np.zeros((200, 500, 3), dtype=np.uint8); loading[:] = (25,25,35)
    cv2.putText(loading, "Extracting SAM3 prompt...", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,180), 2)
    cv2.putText(loading, "Please wait.", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)
    cv2.namedWindow(WIN_LOAD, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_LOAD, 500, 200)
    cv2.imshow(WIN_LOAD, loading); cv2.waitKey(1)

    print(f"Reference box: {box_xywh}")
    print("Extracting SAM3 reference prompt...")

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

    panel   = SliderPanel(1100, params, total_frames=1, fps=30)
    tracker = CentroidTracker(max_disappeared=90, max_distance=200)
    WIN     = "Crystal Detector  |  SAM3  [Screen Capture]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    with autocast_ctx:
        confidence = params[0]['value'] / 100.0
        processor  = Sam3Processor(model, confidence_threshold=confidence)
        w1, h1 = ref_pil.size
        x, y, bw, bh = box_xywh
        box_t    = torch.tensor([x, y, bw, bh], dtype=torch.float32).view(-1, 4)
        norm_box = normalize_bbox(box_xywh_to_cxcywh(box_t), w1, h1).flatten().tolist()
        state1   = processor.set_image(ref_pil)
        state1   = processor._add_box_prompt(box=norm_box, label=True, state=state1)
        state1   = processor._forward_grounding(state1.copy())
        saved_prompt      = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]

        cv2.destroyWindow(WIN_LOAD)
        print("Prompt ready. Select the screen region showing the microscope live view...")

        # ── Step 4: select screen region ──────────────────────────────────────
        region = select_screen_region()
        if region is None: return
        with mss.MSS() as _sct:
            first_frame = cv2.cvtColor(np.array(_sct.grab(region)), cv2.COLOR_BGRA2BGR)
        vid_h, vid_w = first_frame.shape[:2]
        MAX_W  = 1200
        scale  = min(1.0, MAX_W / vid_w)
        disp_w = int(vid_w * scale)
        disp_h = int(vid_h * scale)
        panel.width = disp_w
        cv2.resizeWindow(WIN, disp_w, disp_h + SliderPanel.PANEL_H)
        relay2 = MouseRelay(panel, disp_h)
        cv2.setMouseCallback(WIN, relay2.callback)

        # Background inference thread
        shared = {'frame': None, 'masks': np.zeros((0,),dtype=bool),
                  'boxes': np.zeros((0,4)), 'scores': np.zeros((0,)),
                  'tracked': {}, 'running': True, 'inferring': False}
        lock = threading.Lock()

        def inference_worker():
            while shared['running']:
                with lock:
                    frame = shared['frame']; inferring = shared['inferring']
                if frame is None or inferring:
                    time.sleep(0.01); continue
                with lock: shared['inferring'] = True
                try:
                    conf     = params[0]['value'] / 100.0
                    min_area = params[1]['value']
                    every_n  = params[2]['value']
                    tracker.max_disappeared = params[3]['value']
                    tracker.max_distance    = params[4]['value']
                    processor.confidence_threshold = conf
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    state2 = processor.set_image(frame_pil)
                    state2["prompt"]      = saved_prompt
                    state2["prompt_mask"] = saved_prompt_mask
                    state2 = processor._forward_grounding(state2)
                    masks_t = state2.get("masks"); boxes_t = state2.get("boxes"); scores_t = state2.get("scores")
                    if masks_t is not None and len(masks_t) > 0:
                        masks_np  = masks_t.squeeze(1).cpu().numpy().astype(bool)
                        boxes_np  = boxes_t.cpu().numpy()
                        scores_np = scores_t.cpu().numpy()
                        keep = [i for i,m in enumerate(masks_np) if int(m.sum()) >= min_area]
                        masks_np = masks_np[keep]; boxes_np = boxes_np[keep]; scores_np = scores_np[keep]
                    else:
                        masks_np = np.zeros((0,),dtype=bool); boxes_np = np.zeros((0,4)); scores_np = np.zeros((0,))
                    centroids = [(float((b[0]+b[2])/2), float((b[1]+b[3])/2)) for b in boxes_np]
                    tracked = dict(tracker.update(centroids))
                    with lock:
                        shared['masks']=masks_np; shared['boxes']=boxes_np
                        shared['scores']=scores_np; shared['tracked']=tracked
                except Exception as e:
                    print(f"Inference error: {e}")
                finally:
                    with lock: shared['inferring'] = False

        thread = threading.Thread(target=inference_worker, daemon=True)
        thread.start()

        paused=False; save_count=0; fps_val=0.0; fps_timer=time.time(); fps_count=0
        display = cv2.resize(first_frame, (disp_w, disp_h))
        print("Screen capture detection running. q=quit  SPACE=pause  r=new ref  s=save")
        print(f"Capturing region: {region}")

        with mss.MSS() as sct:
            while True:
                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
                if key in (ord('q'), 27) or window_closed: shared['running']=False; break
                elif key == ord('r'):
                    shared['running']=False
                    cv2.destroyWindow(WIN)
                    run_screen_loop(model, autocast_ctx, ref_image); return
                elif key == ord(' '):       paused=not paused; print("Paused." if paused else "Resumed.")
                elif key == ord('s'):
                    fname=f"crystal_screen_{save_count:04d}.png"
                    combined=np.vstack([display,panel.render()])
                    cv2.imwrite(fname,combined); save_count+=1; print(f"Saved {fname}")

                if not paused:
                    shot  = sct.grab(region)
                    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
                    with lock:
                        shared['frame'] = frame.copy()
                        masks=shared['masks']; boxes=shared['boxes']
                        scores=shared['scores']; tracked=shared['tracked']
                    display = cv2.resize(frame, (disp_w, disp_h))
                    sx, sy  = disp_w/vid_w, disp_h/vid_h
                    if len(masks) > 0:
                        overlay_masks(display, masks, boxes, scores, tracked, (sx, sy))
                    fps_count+=1
                    if fps_count%10==0:
                        now=time.time(); fps_val=10/max(1e-3,now-fps_timer); fps_timer=now

                confidence = params[0]['value']/100.0
                draw_hud(display, len(shared['masks']), tracker.total_seen, fps_val, paused, confidence)
                combined = np.vstack([display, panel.render()])
                cv2.imshow(WIN, combined)

    shared['running'] = False
    cv2.destroyWindow(WIN)
    print(f"Session complete — {tracker.total_seen} unique crystals tracked.")


# ─────────────────────────────────────────────────────────────────────────────
# Camera finder
# ─────────────────────────────────────────────────────────────────────────────
def find_best_camera():
    best_idx, best_px = 0, 0
    found = []
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for i in range(8):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(i, backend)
            except Exception:
                continue
            if not cap.isOpened():
                cap.release(); continue
            ret, _ = cap.read()
            if ret:
                ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append((i, ww, hh))
                if ww * hh > best_px:
                    best_px = ww * hh; best_idx = i
                cap.release()
                break   # found a working backend for this index
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
        cap = None
        for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                _cap = cv2.VideoCapture(camera_index, backend)
            except Exception:
                continue
            if _cap.isOpened():
                ret, _ = _cap.read()
                if ret:
                    cap = _cap
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"Camera {camera_index} opened with backend {backend}")
                    break
            _cap.release()
        if cap is None:
            print(f"ERROR: Could not open camera {camera_index}.")
            return
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
        {'name': 'Confidence',     'min': 5,   'max': 95,    'value': 50,    'step': 1,
         'desc': 'SAM3 score threshold (x0.01). Lower = more detections, may add noise.'},
        {'name': 'Min mask area',  'min': 50,  'max': 50000, 'value': 200,   'step': 50,
         'desc': 'Ignore detections smaller than this many pixels.'},
        {'name': 'Max mask area',  'min': 500, 'max': 500000,'value': 30000, 'step': 500,
         'desc': 'Ignore detections larger than this — removes whole-frame false positives.'},
        {'name': 'NMS IoU thresh', 'min': 5,   'max': 95,    'value': 40,    'step': 5,
         'desc': 'Overlap threshold (x0.01) to merge duplicate boxes.'},
        {'name': 'Process every N','min': 1,   'max': 60,    'value': 3,     'step': 1,
         'desc': 'Run SAM3 every N frames (higher = faster but less smooth).'},
        {'name': 'Memory (frames)','min': 5,   'max': 400,   'value': 120,   'step': 5,
         'desc': 'How long to remember a crystal after it disappears.'},
        {'name': 'Match radius',   'min': 10,  'max': 1000,  'value': 350,   'step': 10,
         'desc': 'Max pixels a crystal can move between frames and still match.'},
    ]

    panel    = SliderPanel(disp_w, params, total_frames=total_frames, fps=video_fps)
    total_h  = disp_h + SliderPanel.PANEL_H
    WIN = "Crystal Detector  |  SAM3"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, disp_w, total_h)

    tracker   = CentroidTracker(max_disappeared=120, max_distance=350)
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
            max_area   = params[2]['value']
            every_n    = params[4]['value']
            memory     = params[5]['value']
            match_r    = params[6]['value']

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

                    # Filter by mask area
                    keep = []
                    for i, m in enumerate(masks_np):
                        area = int(m.sum())
                        if min_area <= area <= max_area:
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

            slider_canvas = panel.render(show_scrubber=(source != 'camera'))
            combined = np.vstack([display, slider_canvas])
            cv2.imshow(WIN, combined)

            # -------- Key handling -------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
            if panel.on_key(key): continue
            if key in (ord('q'), 27) or window_closed:
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
    parser.add_argument("--screen",     action="store_true",        help="Screen capture mode")
    parser.add_argument("--ref",        type=str,   default=None,  help="Separate reference image (optional)")
    parser.add_argument("--checkpoint", type=str,   default=None,  help="Path to sam3.pt checkpoint")
    args = parser.parse_args()

    # If source is forced via CLI flags, skip the menu
    if args.video:
        run(source=args.video, ref_image=args.ref, checkpoint=args.checkpoint)
    elif args.screen:
        device = (torch.device("cuda") if torch.cuda.is_available() else
                  torch.device("mps") if getattr(torch.backends,"mps",None) and
                  torch.backends.mps.is_available() else torch.device("cpu"))
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type=="cuda" else nullcontext())
        model = load_model(args.checkpoint, device)
        run_screen_loop(model, autocast_ctx, ref_image=args.ref)
    elif args.camera is not None:
        run(source='camera', camera_index=args.camera, ref_image=args.ref, checkpoint=args.checkpoint)
    else:
        # Show source selection menu
        source = source_select_screen()
        if source is None:
            sys.exit(0)
        kind, val = source
        if kind == 'screen':
            device = (torch.device("cuda") if torch.cuda.is_available() else
                      torch.device("mps") if getattr(torch.backends,"mps",None) and
                      torch.backends.mps.is_available() else torch.device("cpu"))
            autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if device.type=="cuda" else nullcontext())
            model = load_model(args.checkpoint, device)
            run_screen_loop(model, autocast_ctx, ref_image=args.ref)
        elif kind == 'video':
            run(source=val, ref_image=args.ref, checkpoint=args.checkpoint)
        else:
            idx = find_best_camera()
            if idx is not None:
                run(source='camera', camera_index=idx, ref_image=args.ref, checkpoint=args.checkpoint)
