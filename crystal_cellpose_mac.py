"""
Crystal detector — Cellpose version for macOS (screen capture mode).

Differences from Windows version:
  - Uses MPS (Apple Silicon GPU) or CPU — no CUDA
  - Uses Quartz (CGWindowListCreateImage) for screen capture — avoids mss
    Retina/permission issues on macOS. Falls back to mss if Quartz unavailable.
  - Region selector uses the same drag-to-select OpenCV window

Setup (one time):
    python3 -m venv .venv_mac
    source .venv_mac/bin/activate
    pip install cellpose opencv-python torch torchvision numpy mss
    # Apple Silicon GPU support is included in standard PyTorch for macOS

Usage:
    source .venv_mac/bin/activate
    python crystal_cellpose_mac.py

Keys:
    SPACE  pause/resume    s  save frame    q  quit
    r  reselect screen region

macOS permissions:
    System Settings → Privacy & Security → Screen Recording → allow Terminal/iTerm
"""

import sys, os, time, threading
from collections import OrderedDict

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from cellpose import models

# ── Screen capture backend ────────────────────────────────────────────────────
# Try Quartz first (native macOS, handles Retina correctly), fall back to mss.
try:
    import Quartz
    import Quartz.CoreGraphics as CG
    _USE_QUARTZ = True
except ImportError:
    _USE_QUARTZ = False
    import mss

def _grab_quartz(region):
    """Capture a screen region using Quartz. Returns BGR numpy array."""
    x, y, w, h = region['left'], region['top'], region['width'], region['height']
    rect = CG.CGRectMake(x, y, w, h)
    image = CG.CGWindowListCreateImage(
        rect,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault
    )
    width  = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)
    bpr    = CG.CGImageGetBytesPerRow(image)
    data   = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
    arr    = np.frombuffer(data, dtype=np.uint8).reshape(height, bpr // 4, 4)
    arr    = arr[:, :width, :]          # trim padding bytes
    bgr    = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return bgr

def grab_screen(region, sct=None):
    if _USE_QUARTZ:
        return _grab_quartz(region)
    shot = sct.grab(region)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

def get_full_screenshot():
    """Return full-desktop BGR screenshot for region selector."""
    if _USE_QUARTZ:
        screens = CG.CGDisplayBounds(CG.CGMainDisplayID())
        w = int(screens.size.width); h = int(screens.size.height)
        region = {'left': 0, 'top': 0, 'width': w, 'height': h}
        return _grab_quartz(region), {'left': 0, 'top': 0}
    import mss as _mss
    with _mss.MSS() as sct:
        monitor = sct.monitors[0]
        shot = sct.grab(monitor)
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR), monitor


# ─────────────────────────────────────────────────────────────────────────────
# Centroid tracker (velocity + graveyard)
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
# Slider panel (full version with −/+ buttons and click-to-type)
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

    def __init__(self, width, params):
        self.width    = width
        self.params   = params
        self.dragging = None
        self.edit_idx = None
        self.edit_buf = ""

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
        x1t = self.width - self.PAD_R; bx = x1t + 8; ry = self._row_y(idx)
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

    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._commit_edit()
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
            if self.dragging is not None:
                self.params[self.dragging]['value']=self._x_to_val(self.dragging,x)
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

    def render(self):
        c=np.zeros((self.PANEL_H,self.width,3),dtype=np.uint8); c[:]=(25,25,35)
        cv2.rectangle(c,(0,0),(self.width,self.SCRUB_H),(18,18,28),-1)
        cv2.putText(c,"CELLPOSE — LIVE SCREEN CAPTURE  [macOS]",(10,self.SCRUB_H//2+6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,120,150),1)
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
            disp=(self.edit_buf+"|") if editing else (f"{p['value']:.2f}" if isinstance(p['value'],float) else str(p['value']))
            cv2.putText(c,disp,(vx0+4,vy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,180),1)
            px0,py0,px1,py1=self._btn_plus_rect(i)
            cv2.rectangle(c,(px0,py0),(px1,py1),(60,60,80),-1); cv2.rectangle(c,(px0,py0),(px1,py1),(140,140,160),1)
            cv2.putText(c,"+",(px0+5,py1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,220,100),1)
        cv2.putText(c,"q=quit  r=reselect region  s=save  SPACE=pause/resume  (click value box to type, ENTER to confirm)",
                    (10,self.PANEL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,100,120),1)
        return c


# ─────────────────────────────────────────────────────────────────────────────
# Screen region selector
# ─────────────────────────────────────────────────────────────────────────────
def select_screen_region():
    full_bgr, monitor = get_full_screenshot()
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
        cv2.putText(canvas,"Drag a box over the microscope live view, then press ENTER",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
        cv2.putText(canvas,"Drag a box over the microscope live view, then press ENTER",
                    (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),1)
        if state['pt1'] and state['pt2']:
            col=(0,255,255) if state['drawing'] else (0,200,255)
            cv2.rectangle(canvas,state['pt1'],state['pt2'],col,2)
        cv2.imshow(WIN,canvas)
        key=cv2.waitKey(30)&0xFF
        if key in (ord('q'),27): cv2.destroyWindow(WIN); return None
        if key in (13,ord(' ')) and state['done'] and state['pt1'] and state['pt2']:
            x1=min(state['pt1'][0],state['pt2'][0]); y1=min(state['pt1'][1],state['pt2'][1])
            x2=max(state['pt1'][0],state['pt2'][0]); y2=max(state['pt1'][1],state['pt2'][1])
            if (x2-x1)<10 or (y2-y1)<10: continue
            inv=1.0/scale
            region={
                'left':  int(x1*inv) + monitor.get('left', 0),
                'top':   int(y1*inv) + monitor.get('top',  0),
                'width': int((x2-x1)*inv),
                'height':int((y2-y1)*inv),
            }
            cv2.destroyWindow(WIN)
            print(f"Capture region: {region}")
            return region


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [(0,200,100),(255,120,0),(0,160,255),(220,0,220),(0,220,220),(220,200,0),
          (255,255,0),(0,255,255),(255,0,128),(128,255,0)]

def overlay_masks(display, masks, tracked):
    for i, mask in enumerate(masks):
        col = COLORS[i % len(COLORS)]
        ov = display.copy()
        ov[mask] = (ov[mask].astype(float)*0.4 + np.array(col,float)*0.6).astype(np.uint8)
        cv2.addWeighted(ov,0.65,display,0.35,0,display)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, col, 2)
    for oid,(cx,cy) in tracked.items():
        cv2.circle(display,(int(cx),int(cy)),5,(0,0,255),-1)
        cv2.putText(display,f"#{oid}",(int(cx)+7,int(cy)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,230,0),1)

def draw_hud(display, n, total, fps, paused, diameter, model_type, device):
    hud=display.copy(); cv2.rectangle(hud,(6,6),(460,155),(0,0,0),-1)
    cv2.addWeighted(hud,0.45,display,0.55,0,display)
    cv2.putText(display,f"CRYSTAL DETECTOR  [Cellpose {model_type} | {device}]",(14,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.putText(display,f"In frame:   {n:3d}{'  [PAUSED]' if paused else ''}",(14,58),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,180),2)
    cv2.putText(display,f"Total seen: {total:3d}",(14,84),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,255),1)
    cv2.putText(display,f"FPS: {fps:.1f}   diameter={diameter}px",(14,108),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,160,160),1)
    cv2.putText(display,"r=reselect  s=save  SPACE=pause  q=quit",(14,132),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(120,120,140),1)


# ─────────────────────────────────────────────────────────────────────────────
# Cellpose inference
# ─────────────────────────────────────────────────────────────────────────────
def masks_from_cellpose(model, frame_rgb, diameter, flow_threshold, cellprob_threshold,
                        min_area, max_area):
    masks_labeled, _, _ = model.eval(
        frame_rgb,
        diameter=diameter if diameter > 0 else None,
        channels=[0, 0],
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        batch_size=1,
    )
    masks = []
    for label in np.unique(masks_labeled):
        if label == 0: continue
        m = masks_labeled == label
        area = int(m.sum())
        if min_area <= area <= max_area:
            masks.append(m)
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────
def detection_loop(model, model_type, device_name, region, params, tracker):
    rh, rw = region['height'], region['width']
    scale  = min(1.0, 1100/rw)
    dw, dh = int(rw*scale), int(rh*scale)

    panel = SliderPanel(dw, params)
    WIN   = "Crystal Detector — Cellpose [macOS]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh + SliderPanel.PANEL_H)

    class Relay:
        def __init__(self,p,vh): self.p=p; self.vh=vh
        def cb(self,e,x,y,f,_):
            if y >= self.vh: self.p.on_mouse(e,x,y-self.vh)
    cv2.setMouseCallback(WIN, Relay(panel,dh).cb)

    shared = {'frame': None, 'masks': [], 'tracked': {},
              'running': True, 'inferring': False}
    lock = threading.Lock()

    def inference_worker():
        while shared['running']:
            with lock:
                frame = shared['frame']; inferring = shared['inferring']
            if frame is None or inferring:
                time.sleep(0.02); continue
            with lock: shared['inferring'] = True
            try:
                diameter        = params[0]['value']
                flow_threshold  = params[1]['value'] / 100.0
                cellprob_thresh = params[2]['value'] / 10.0 - 6.0
                min_area        = params[3]['value']
                max_area        = params[4]['value']
                tracker.max_disappeared = params[5]['value']
                tracker.max_distance    = params[6]['value']

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                masks = masks_from_cellpose(model, frame_rgb, diameter,
                                            flow_threshold, cellprob_thresh,
                                            min_area, max_area)
                centroids = []
                for m in masks:
                    ys, xs = np.where(m)
                    centroids.append((float(xs.mean()), float(ys.mean())))
                tracked = dict(tracker.update(centroids))
                with lock:
                    shared['masks']   = masks
                    shared['tracked'] = tracked
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                with lock: shared['inferring'] = False

    thread = threading.Thread(target=inference_worker, daemon=True)
    thread.start()

    paused=False; save_count=0; fps_val=0.0; fps_timer=time.time(); fps_count=0
    display=np.zeros((dh,dw,3),dtype=np.uint8)
    print("Detection running. q=quit  SPACE=pause  r=reselect region  s=save")

    # Keep mss context alive for fallback
    sct_ctx = mss.MSS() if not _USE_QUARTZ else None

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
            if panel.on_key(key): pass
            elif key in (ord('q'),27) or window_closed:
                shared['running']=False; break
            elif key==ord('r'):
                shared['running']=False; return 'reset'
            elif key==ord(' '):
                paused=not paused; print("Paused." if paused else "Resumed.")
            elif key==ord('s'):
                fname=f"crystal_cellpose_{save_count:04d}.png"
                combined=np.vstack([display,panel.render()])
                cv2.imwrite(fname,combined); save_count+=1; print(f"Saved {fname}")

            if not paused:
                frame = grab_screen(region, sct_ctx)
                with lock:
                    shared['frame'] = frame.copy()
                    masks   = shared['masks']
                    tracked = shared['tracked']

                display = cv2.resize(frame,(dw,dh))
                if masks:
                    scaled = [cv2.resize(m.astype(np.uint8),(dw,dh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
                              for m in masks]
                    sx,sy = dw/rw, dh/rh
                    scaled_tracked = {oid:(cx*sx,cy*sy) for oid,(cx,cy) in tracked.items()}
                    overlay_masks(display, scaled, scaled_tracked)

                fps_count+=1
                if fps_count%10==0:
                    now=time.time(); fps_val=10/max(1e-3,now-fps_timer); fps_timer=now

            diameter = params[0]['value']
            draw_hud(display,len(shared['masks']),tracker.total_seen,
                     fps_val,paused,diameter,model_type,device_name)
            combined=np.vstack([display,panel.render()])
            cv2.imshow(WIN,combined)
    finally:
        if sct_ctx: sct_ctx.close()
        shared['running']=False
        cv2.destroyWindow(WIN)

    return 'quit'


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run():
    # Device selection: MPS (Apple Silicon) > CUDA (unlikely on Mac) > CPU
    if torch.backends.mps.is_available():
        device_name = 'mps'
        use_gpu = True
    elif torch.cuda.is_available():
        device_name = 'cuda'
        use_gpu = True
    else:
        device_name = 'cpu'
        use_gpu = False
    print(f"Device: {device_name}")

    model_type = 'cyto3'
    print(f"Loading Cellpose model: {model_type} ...")
    model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    print("Model loaded.")

    if _USE_QUARTZ:
        print("Screen capture: Quartz (native macOS)")
    else:
        print("Screen capture: mss (fallback — install pyobjc-framework-Quartz for better Retina support)")

    params = [
        {'name': 'Diameter (px)',    'min': 0,   'max': 500,   'value': 30,    'step': 1,
         'desc': 'Expected crystal diameter in pixels. Set 0 to auto-estimate.'},
        {'name': 'Flow threshold',   'min': 0,   'max': 100,   'value': 40,    'step': 1,
         'desc': 'Flow error threshold (x0.01). Lower = stricter, fewer false positives.'},
        {'name': 'Cell prob (x10+6)','min': 0,   'max': 60,    'value': 30,    'step': 1,
         'desc': 'Cellpose probability threshold mapped to -6..0. Higher = stricter.'},
        {'name': 'Min mask area',    'min': 50,  'max': 50000, 'value': 200,   'step': 50,
         'desc': 'Ignore detections smaller than this many pixels.'},
        {'name': 'Max mask area',    'min': 500, 'max': 500000,'value': 30000, 'step': 500,
         'desc': 'Ignore detections larger than this.'},
        {'name': 'Memory (frames)',  'min': 5,   'max': 400,   'value': 120,   'step': 5,
         'desc': 'How long tracker remembers a disappeared crystal.'},
        {'name': 'Match radius',     'min': 10,  'max': 1000,  'value': 350,   'step': 10,
         'desc': 'Max pixels a crystal can move and still match.'},
    ]

    tracker = CentroidTracker()

    while True:
        print("Select the screen region showing the microscope live view...")
        region = select_screen_region()
        if region is None: break

        tracker.reset()
        result = detection_loop(model, model_type, device_name, region, params, tracker)
        print(f"Session: {tracker.total_seen} unique crystals tracked.")
        if result == 'quit': break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    run()
