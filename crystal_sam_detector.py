"""
Real-time crystal detector — SAM3, dual-reference GUI.

Workflow:
  1. A file-picker opens — select your FIRST reference image.
  2. Draw a box on it around one crystal → ENTER to confirm.
  3. A file-picker opens — select your SECOND reference image.
  4. Draw a box on it → ENTER to confirm.
  5. Choose source: [L] live camera  or  [V] recorded video file.
  6. SAM3 runs BOTH prompts on every frame and counts ALL matching crystals.

Usage:
    python crystal_sam_detector.py                     # full interactive flow
    python crystal_sam_detector.py --video path.mp4    # skip source-select
    python crystal_sam_detector.py --camera 0          # skip source-select
    python crystal_sam_detector.py --ref1 a.png --ref2 b.png  # skip file-picker

Keys (box drawing):
    click+drag  draw box    ENTER/SPACE confirm    r redraw    q quit

Keys (detection):
    SPACE  pause/resume    r  redo references    s  save frame    q  quit
"""

import sys, os, time, argparse
from collections import OrderedDict
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image

# ── tkinter file dialog ───────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from importlib.resources import files as importlib_files
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox

DEFAULT_CKPT   = os.path.join(PROJECT_ROOT, "sam3.pt")
DEFAULT_FOLDER = PROJECT_ROOT          # file-picker opens here


# ─────────────────────────────────────────────────────────────────────────────
# File picker (uses native Windows dialog)
# ─────────────────────────────────────────────────────────────────────────────
def pick_images(title="Select reference images", initial_dir=DEFAULT_FOLDER):
    """Open a native file dialog allowing MULTIPLE image selections. Returns list of paths."""
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title=title,
        initialdir=initial_dir,
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All files", "*.*")]
    )
    root.destroy()
    return list(paths) if paths else []


def pick_video(title="Select video file", initial_dir=DEFAULT_FOLDER):
    """Open a native file dialog and return the chosen video path, or None."""
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None


# ─────────────────────────────────────────────────────────────────────────────
# Centroid tracker
# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=120, max_distance=300):
        self.next_id     = 0
        self.objects     = OrderedDict()   # id -> current centroid
        self.velocity    = OrderedDict()   # id -> (vx, vy)  — estimated pixels/update
        self.disappeared = OrderedDict()
        self.graveyard   = []              # (centroid, velocity, age)
        self.graveyard_ttl   = 400
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.total_seen      = 0

    # ── predict where each object will be next update ────────────────────────
    def _predicted(self, oid):
        cx, cy = self.objects[oid]
        vx, vy = self.velocity.get(oid, (0, 0))
        age = self.disappeared[oid]          # extrapolate further if unseen longer
        return cx + vx * (age + 1), cy + vy * (age + 1)

    def register(self, centroid, vel=(0, 0)):
        # Check graveyard — same crystal reappeared nearby?
        best_dist, best_idx = float('inf'), -1
        for i, (gc, gv, _) in enumerate(self.graveyard):
            d = np.hypot(centroid[0]-gc[0], centroid[1]-gc[1])
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_dist < self.max_distance * 1.5 and best_idx >= 0:
            _, gv, _ = self.graveyard.pop(best_idx)
            vel = gv   # inherit last known velocity
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

        # Use PREDICTED positions for matching (velocity-aware)
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
            # Smooth velocity with exponential moving average
            old_vx, old_vy = self.velocity[oid]
            alpha = 0.5
            self.velocity[oid]   = (alpha*(new_cx-old_cx) + (1-alpha)*old_vx,
                                    alpha*(new_cy-old_cy) + (1-alpha)*old_vy)
            self.objects[oid]    = centroids[c]
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
# Slider panel
# ─────────────────────────────────────────────────────────────────────────────
class SliderPanel:
    PANEL_H  = 460
    PAD_L    = 210          # left edge of slider track
    PAD_R    = 200          # right margin (room for  −  value  +)
    ROW_H    = 36
    TRACK_H  = 6
    HANDLE_R = 10
    SCRUB_H  = 52
    SCRUB_PAD = 10
    BTN_W    = 26           # width of − / + buttons
    BTN_H    = 22

    def __init__(self, width, params, total_frames=0, fps=30):
        self.width  = width
        self.params = params
        self.dragging   = None      # 'scrub' | param-index | None
        self.edit_idx   = None      # param being typed into
        self.edit_buf   = ""        # keyboard buffer while typing
        self.total_frames = max(1, total_frames)
        self.fps          = max(1, fps)
        self.current_frame = 0
        self.seek_frame    = None

    # ── geometry ──────────────────────────────────────────────────────────────
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
        """(x0,y0,x1,y1) for the − button."""
        x1_track = self.width - self.PAD_R
        bx = x1_track + 8
        ry = self._row_y(idx)
        return bx, ry - self.BTN_H//2, bx + self.BTN_W, ry + self.BTN_H//2

    def _btn_plus_rect(self, idx):
        bx = self.width - self.PAD_R + 8 + self.BTN_W + 52 + 8
        ry = self._row_y(idx)
        return bx, ry - self.BTN_H//2, bx + self.BTN_W, ry + self.BTN_H//2

    def _val_box_rect(self, idx):
        """Clickable value display box between − and +."""
        bx0 = self.width - self.PAD_R + 8 + self.BTN_W + 4
        bx1 = bx0 + 52
        ry  = self._row_y(idx)
        return bx0, ry - self.BTN_H//2, bx1, ry + self.BTN_H//2

    def _clamp(self, idx, val):
        p = self.params[idx]
        s = p.get('step', 1)
        val = max(p['min'], min(p['max'], val))
        return int(round(val/s)*s) if s >= 1 else float(f"{val:.3f}")

    # ── scrubber helpers ──────────────────────────────────────────────────────
    def _scrub_x(self):
        pad = self.SCRUB_PAD; x0, x1 = pad, self.width-pad
        return int(x0 + self.current_frame/self.total_frames*(x1-x0)), x0, x1

    def _x_to_frame(self, x):
        pad = self.SCRUB_PAD; x0, x1 = pad, self.width-pad
        return int(np.clip((x-x0)/max(1,x1-x0),0,1)*self.total_frames)

    # ── mouse ─────────────────────────────────────────────────────────────────
    def on_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Cancel any active text edit
            self._commit_edit()

            # Scrubber
            hx, sx0, sx1 = self._scrub_x()
            if sx0<=x<=sx1 and abs(y-self.SCRUB_H//2)<20:
                self.dragging='scrub'; self.seek_frame=self._x_to_frame(x); return

            for i in range(len(self.params)):
                ry = self._row_y(i)

                # − button
                mx0,my0,mx1,my1 = self._btn_minus_rect(i)
                if mx0<=x<=mx1 and my0<=y<=my1:
                    self.params[i]['value'] = self._clamp(i, self.params[i]['value'] - self.params[i].get('step',1))
                    return

                # + button
                px0,py0,px1,py1 = self._btn_plus_rect(i)
                if px0<=x<=px1 and py0<=y<=py1:
                    self.params[i]['value'] = self._clamp(i, self.params[i]['value'] + self.params[i].get('step',1))
                    return

                # Value box → enter text edit mode
                vx0,vy0,vx1,vy1 = self._val_box_rect(i)
                if vx0<=x<=vx1 and vy0<=y<=vy1:
                    self.edit_idx = i
                    self.edit_buf = str(self.params[i]['value'])
                    return

                # Slider track / handle
                hx2 = self._val_to_x(i); x0t, x1t = self._tx(i)
                if abs(x-hx2)<self.HANDLE_R+8 and abs(y-ry)<self.HANDLE_R+8:
                    self.dragging=i; return
                if x0t<=x<=x1t and abs(y-ry)<16:
                    self.params[i]['value']=self._x_to_val(i,x); self.dragging=i; return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging=='scrub': self.seek_frame=self._x_to_frame(x)
            elif self.dragging is not None:
                self.params[self.dragging]['value']=self._x_to_val(self.dragging,x)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging=None

    # ── keyboard (call from main loop when edit_idx is set) ───────────────────
    def on_key(self, key):
        """Pass raw cv2.waitKey() result. Returns True if key was consumed."""
        if self.edit_idx is None:
            return False
        if key in (13, 10):                     # ENTER — commit
            self._commit_edit(); return True
        if key == 27:                           # ESC — cancel
            self.edit_idx = None; self.edit_buf = ""; return True
        if key == 8:                            # BACKSPACE
            self.edit_buf = self.edit_buf[:-1]; return True
        if 48 <= key <= 57:                     # digit 0-9
            self.edit_buf += chr(key); return True
        if key == ord('.') and '.' not in self.edit_buf:
            self.edit_buf += '.'; return True
        if key == ord('-') and self.edit_buf == "":
            self.edit_buf = '-'; return True
        return True                             # swallow all keys while editing

    def _commit_edit(self):
        if self.edit_idx is None: return
        try:
            val = float(self.edit_buf)
            self.params[self.edit_idx]['value'] = self._clamp(self.edit_idx, val)
        except ValueError:
            pass
        self.edit_idx = None; self.edit_buf = ""

    # ── render ────────────────────────────────────────────────────────────────
    def render(self, show_scrubber=True):
        c = np.zeros((self.PANEL_H, self.width, 3), dtype=np.uint8); c[:] = (25,25,35)

        # Scrubber
        if show_scrubber:
            cv2.rectangle(c,(0,0),(self.width,self.SCRUB_H),(18,18,28),-1)
            cv2.putText(c,"VIDEO POSITION",(10,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(120,120,150),1)
            pad=self.SCRUB_PAD; sx0,sx1=pad,self.width-pad; sy=self.SCRUB_H//2+5
            cv2.line(c,(sx0,sy),(sx1,sy),(55,55,70),4)
            hx,_,_=self._scrub_x(); cv2.line(c,(sx0,sy),(hx,sy),(80,140,255),4)
            col=(140,200,255) if self.dragging=='scrub' else (100,170,255)
            cv2.circle(c,(hx,sy),10,col,-1); cv2.circle(c,(hx,sy),10,(255,255,255),1)
            secs=int(self.current_frame/self.fps); tot=int(self.total_frames/self.fps)
            cv2.putText(c,f"{secs//60}:{secs%60:02d}",(sx0,sy-14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,200,255),1)
            cv2.putText(c,f"{tot//60}:{tot%60:02d}",(sx1-35,sy-14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,130),1)
        cv2.line(c,(0,self.SCRUB_H),(self.width,self.SCRUB_H),(45,45,60),1)

        # Header
        cv2.rectangle(c,(0,self.SCRUB_H),(self.width,self.SCRUB_H+22),(40,40,55),-1)
        cv2.putText(c,"PARAMETERS — drag slider  |  click value to type  |  use  −  +  buttons",
                    (10,self.SCRUB_H+16),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,200),1)

        for i,p in enumerate(self.params):
            ry   = self._row_y(i)
            x0t, x1t = self._tx(i)
            hx2  = self._val_to_x(i)
            editing = (self.edit_idx == i)

            # Description + name
            cv2.putText(c,p['desc'],(x0t,ry-12),cv2.FONT_HERSHEY_SIMPLEX,0.31,(120,120,140),1)
            cv2.putText(c,p['name'],(8,ry+4),cv2.FONT_HERSHEY_SIMPLEX,0.44,(210,210,230),1)

            # Track
            cv2.line(c,(x0t,ry),(x1t,ry),(60,60,75),self.TRACK_H)
            cv2.line(c,(x0t,ry),(hx2,ry),(0,180,120),self.TRACK_H)

            # Handle
            hcol = (0,255,180) if self.dragging==i else (0,210,140)
            cv2.circle(c,(hx2,ry),self.HANDLE_R,hcol,-1)
            cv2.circle(c,(hx2,ry),self.HANDLE_R,(255,255,255),1)

            # − button
            mx0,my0,mx1,my1 = self._btn_minus_rect(i)
            cv2.rectangle(c,(mx0,my0),(mx1,my1),(60,60,80),-1)
            cv2.rectangle(c,(mx0,my0),(mx1,my1),(140,140,160),1)
            cv2.putText(c,"-",(mx0+6,my1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(220,100,100),1)

            # Value box
            vx0,vy0,vx1,vy1 = self._val_box_rect(i)
            bg = (50,50,80) if editing else (35,35,55)
            cv2.rectangle(c,(vx0,vy0),(vx1,vy1),bg,-1)
            border = (0,200,255) if editing else (100,100,130)
            cv2.rectangle(c,(vx0,vy0),(vx1,vy1),border,1)
            disp = (self.edit_buf + "|") if editing else (
                f"{p['value']:.1f}" if isinstance(p['value'],float) else str(p['value']))
            cv2.putText(c,disp,(vx0+4,vy1-4),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,255,180),1)

            # + button
            px0,py0,px1,py1 = self._btn_plus_rect(i)
            cv2.rectangle(c,(px0,py0),(px1,py1),(60,60,80),-1)
            cv2.rectangle(c,(px0,py0),(px1,py1),(140,140,160),1)
            cv2.putText(c,"+",(px0+5,py1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(100,220,100),1)

        cv2.putText(c,"q=quit  r=new refs  s=save  SPACE=pause/resume  (click value box to type, ENTER to confirm)",
                    (10,self.PANEL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,100,120),1)
        return c


# ─────────────────────────────────────────────────────────────────────────────
# Box drawer
# ─────────────────────────────────────────────────────────────────────────────
class BoxDrawer:
    def __init__(self):
        self.pt1=self.pt2=None; self.drawing=self.confirmed=False

    def on_mouse(self,event,x,y):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.pt1=(x,y);self.pt2=(x,y);self.drawing=True;self.confirmed=False
        elif event==cv2.EVENT_MOUSEMOVE and self.drawing: self.pt2=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and self.drawing:
            self.pt2=(x,y);self.drawing=False;self.confirmed=True

    def reset(self): self.pt1=self.pt2=None;self.drawing=self.confirmed=False

    @property
    def box(self):
        if not self.pt1 or not self.pt2: return None
        x1,y1=min(self.pt1[0],self.pt2[0]),min(self.pt1[1],self.pt2[1])
        x2,y2=max(self.pt1[0],self.pt2[0]),max(self.pt1[1],self.pt2[1])
        return (x1,y1,x2-x1,y2-y1) if x2-x1>5 and y2-y1>5 else None

    def draw_on(self,img):
        box=self.box
        if not box: return
        x,y,w,h=box; col=(0,255,255) if self.drawing else (255,200,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),col,2)
        cv2.putText(img,"drawing..." if self.drawing else "REFERENCE (ENTER=confirm)",
                    (x+4,max(16,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)


# ─────────────────────────────────────────────────────────────────────────────
# Setup screen: show image, user draws reference box
# ─────────────────────────────────────────────────────────────────────────────
def setup_screen(ref_path, label):
    """Returns (PIL image, (x,y,w,h)) or (None, None) on quit."""
    base = cv2.imread(ref_path)
    if base is None:
        print(f"Cannot read: {ref_path}"); return None, None
    h, w = base.shape[:2]
    scale = min(1.0, 1100/w)
    dw, dh = int(w*scale), int(h*scale)
    small = cv2.resize(base, (dw, dh))
    drawer = BoxDrawer()
    WIN = f"Reference {label}: {os.path.basename(ref_path)}  —  draw box then ENTER"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, dw, dh)
    cv2.setMouseCallback(WIN, lambda e,x,y,f,p: drawer.on_mouse(e,x,y))
    while True:
        canvas = small.copy(); drawer.draw_on(canvas)
        for li, line in enumerate([
            f"Ref {label}: {os.path.basename(ref_path)}",
            "Draw box around ONE crystal  |  ENTER=confirm  r=redraw  q=quit"
        ]):
            cv2.putText(canvas,line,(12,24+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,0,0),3)
            cv2.putText(canvas,line,(12,24+li*22),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27): cv2.destroyWindow(WIN); return None, None
        if key == ord('r'): drawer.reset()
        if key in (13, ord(' ')):
            box = drawer.box
            if box:
                inv = 1.0/scale
                x,y,bw,bh = (int(v*inv) for v in box)
                cv2.destroyWindow(WIN)
                return Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)), (x,y,bw,bh)
            else:
                cv2.putText(canvas,"Draw a box first!",(dw//2-120,dh//2),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,50,255),2)
                cv2.imshow(WIN,canvas); cv2.waitKey(600)


# ─────────────────────────────────────────────────────────────────────────────
# Source selection screen
# ─────────────────────────────────────────────────────────────────────────────
def source_select_screen(force_video=None, force_camera=None):
    """Returns ('video', path) | ('camera', idx) | None."""
    if force_video:   return ('video',  force_video)
    if force_camera is not None: return ('camera', force_camera)

    WIN = "Select Source"
    canvas = np.zeros((320, 640, 3), dtype=np.uint8); canvas[:] = (25,25,35)
    items = [
        ("CRYSTAL DETECTOR — SAM3",       (255,255,255), 0.75, 2),
        ("",None,0,0),
        ("L  —  Live camera",              (0,255,180),  0.62, 1),
        ("V  —  Video file  (file picker)",(0,200,255),  0.62, 1),
        ("Q  —  Quit",                     (180,80,80),  0.5,  1),
    ]
    y=55
    for text,col,sc,th in items:
        if col: cv2.putText(canvas,text,(40,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col,th)
        y+=50
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN,640,320)
    cv2.imshow(WIN,canvas)
    while True:
        key=cv2.waitKey(50)&0xFF
        if key in(ord('q'),27): cv2.destroyWindow(WIN); return None
        if key==ord('l'): cv2.destroyWindow(WIN); return ('camera',0)
        if key==ord('v'):
            cv2.destroyWindow(WIN)
            path = pick_video("Select video file")
            if path and os.path.exists(path): return ('video', path)
            print("No video selected."); return None


# ─────────────────────────────────────────────────────────────────────────────
# Extract SAM3 prompt
# ─────────────────────────────────────────────────────────────────────────────
def extract_prompt(processor, pil_img, box_xywh):
    w1,h1=pil_img.size; x,y,bw,bh=box_xywh
    box_t=torch.tensor([x,y,bw,bh],dtype=torch.float32).view(-1,4)
    norm_box=normalize_bbox(box_xywh_to_cxcywh(box_t),w1,h1).flatten().tolist()
    state=processor.set_image(pil_img)
    state=processor._add_box_prompt(box=norm_box,label=True,state=state)
    state=processor._forward_grounding(state.copy())
    return state["prompt"], state["prompt_mask"]


# ─────────────────────────────────────────────────────────────────────────────
# NMS — remove overlapping duplicate boxes (IoU threshold)
# ─────────────────────────────────────────────────────────────────────────────
def nms(boxes, scores, iou_thresh=0.4):
    """boxes: (N,4) x0y0x1y1,  scores: (N,).  Returns keep indices."""
    if len(boxes) == 0:
        return []
    x0,y0,x1,y1 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x1-x0)*(y1-y0)
    order = scores.argsort()[::-1]
    keep  = []
    while len(order):
        i = order[0]; keep.append(i)
        ix0 = np.maximum(x0[i], x0[order[1:]])
        iy0 = np.maximum(y0[i], y0[order[1:]])
        ix1 = np.minimum(x1[i], x1[order[1:]])
        iy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0, ix1-ix0) * np.maximum(0, iy1-iy0)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Run SAM3 with both prompts, merge + NMS
# ─────────────────────────────────────────────────────────────────────────────
def run_sam3_dual(processor, prompts, pil_img, confidence, min_area, max_area, nms_iou=0.4):
    all_masks, all_boxes, all_scores = [], [], []
    processor.confidence_threshold = confidence

    # Encode image ONCE — reuse backbone features for both prompts
    base_state = processor.set_image(pil_img)

    for (prompt, prompt_mask) in prompts:
        state = dict(base_state)            # shallow copy — backbone_out shared, not re-run
        state["prompt"]      = prompt
        state["prompt_mask"] = prompt_mask
        state = processor._forward_grounding(state)
        masks_t=state.get("masks"); boxes_t=state.get("boxes"); scores_t=state.get("scores")
        if masks_t is None or len(masks_t)==0: continue
        masks_np  = masks_t.squeeze(1).cpu().float().numpy().astype(bool)
        boxes_np  = boxes_t.cpu().float().numpy()
        scores_np = scores_t.cpu().float().numpy()
        for i,m in enumerate(masks_np):
            area = int(m.sum())
            if min_area <= area <= max_area:
                all_masks.append(m); all_boxes.append(boxes_np[i]); all_scores.append(scores_np[i])
    if not all_masks:
        return np.zeros((0,),dtype=bool), np.zeros((0,4)), np.zeros((0,))
    masks_a  = np.stack(all_masks)
    boxes_a  = np.stack(all_boxes)
    scores_a = np.array(all_scores)
    # NMS to remove overlapping duplicates from both prompts
    keep = nms(boxes_a, scores_a, iou_thresh=nms_iou)
    return masks_a[keep], boxes_a[keep], scores_a[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [(0,200,100),(255,120,0),(0,160,255),(220,0,220),(0,220,220),(220,200,0)]

def overlay_detections(display, masks_np, boxes_np, scores_np, tracked, sx, sy):
    for i in range(len(masks_np)):
        col=COLORS[i%len(COLORS)]
        bx0,by0,bx1,by1=(int(v) for v in boxes_np[i])
        bx0d,by0d,bx1d,by1d=int(bx0*sx),int(by0*sy),int(bx1*sx),int(by1*sy)
        mask_disp=cv2.resize(masks_np[i].astype(np.uint8),(display.shape[1],display.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        ov=display.copy()
        ov[mask_disp]=(ov[mask_disp].astype(float)*0.4+np.array(col,float)*0.6).astype(np.uint8)
        cv2.addWeighted(ov,0.65,display,0.35,0,display)
        cv2.rectangle(display,(bx0d,by0d),(bx1d,by1d),col,2)
        cv2.putText(display,f"{scores_np[i]:.2f}",(bx0d+4,by0d+16),cv2.FONT_HERSHEY_SIMPLEX,0.44,col,1)
    for oid,(cx,cy) in tracked.items():
        cxd,cyd=int(cx*sx),int(cy*sy)
        cv2.circle(display,(cxd,cyd),5,(0,0,255),-1)
        cv2.putText(display,f"#{oid}",(cxd+7,cyd-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,230,0),1)

def draw_hud(display,n,total,fps,paused,conf,label):
    hud=display.copy(); cv2.rectangle(hud,(6,6),(370,150),(0,0,0),-1)
    cv2.addWeighted(hud,0.45,display,0.55,0,display)
    cv2.putText(display,f"CRYSTAL DETECTOR  [{label}]",(14,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    cv2.putText(display,f"In frame:   {n:3d}{'  [PAUSED]' if paused else ''}",(14,58),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,180),2)
    cv2.putText(display,f"Total seen: {total:3d}",(14,84),cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,255),1)
    cv2.putText(display,f"FPS: {fps:.1f}   conf>={conf:.2f}",(14,108),cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,160,160),1)
    cv2.putText(display,"r=new ref  s=save  SPACE=pause  q=quit",(14,130),cv2.FONT_HERSHEY_SIMPLEX,0.38,(120,120,140),1)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter tuning screen — run on first frame before playback starts
# ─────────────────────────────────────────────────────────────────────────────
def param_tuning_screen(processor, prompts, first_frame_bgr, params, panel, WIN):
    """
    Shows the first frame with live detections. User adjusts sliders until
    happy, then presses ENTER or SPACE to start playback. Returns False to quit.
    """
    vh, vw = first_frame_bgr.shape[:2]
    scale = min(1.0, 1100/vw)
    dw, dh = int(vw*scale), int(vh*scale)
    panel.width = dw
    cv2.resizeWindow(WIN, dw, dh + SliderPanel.PANEL_H)

    class Relay:
        def __init__(self,p,vh): self.p=p; self.vh=vh
        def cb(self,e,x,y,f,_):
            if y-self.vh>=0: self.p.on_mouse(e,x,y-self.vh)
    cv2.setMouseCallback(WIN, Relay(panel, dh).cb)

    last_masks = np.zeros((0,),dtype=bool)
    last_boxes = np.zeros((0,4)); last_scores = np.zeros((0,))
    needs_rerun = True
    sx, sy = dw/vw, dh/vh

    print("Parameter tuning screen — adjust sliders, press ENTER to start.")

    while True:
        confidence = params[0]['value'] / 100.0
        min_area   = params[1]['value']
        max_area   = params[2]['value']
        nms_iou    = params[3]['value'] / 100.0

        if needs_rerun:
            fh, fw = first_frame_bgr.shape[:2]
            sam_w = min(fw, 640); sam_h = int(fh * sam_w / fw)
            small = cv2.resize(first_frame_bgr, (sam_w, sam_h))
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            bxs = fw / sam_w;  bys = fh / sam_h
            last_masks, last_boxes, last_scores = run_sam3_dual(
                processor, prompts, pil_img, confidence, min_area, max_area, nms_iou)
            if len(last_boxes):
                last_boxes = last_boxes * np.array([bxs, bys, bxs, bys])
                last_masks = np.stack([
                    cv2.resize(m.astype(np.uint8), (fw, fh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in last_masks])
            needs_rerun = False

        display = cv2.resize(first_frame_bgr, (dw, dh))
        if len(last_masks) > 0:
            overlay_detections(display, last_masks, last_boxes, last_scores, {}, sx, sy)

        # Banner
        hud = display.copy(); cv2.rectangle(hud,(6,6),(600,110),(0,0,0),-1)
        cv2.addWeighted(hud,0.45,display,0.55,0,display)
        cv2.putText(display,"PARAMETER TUNING — first frame preview",(14,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,0),2)
        cv2.putText(display,f"Detections: {len(last_masks)}   Adjust sliders below, then press ENTER to start",(14,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(200,200,255),1)
        cv2.putText(display,"ENTER/SPACE = start video    q = quit    r = new references",(14,86),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(120,200,120),1)

        combined = np.vstack([display, panel.render(show_scrubber=False)])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(30) & 0xFF
        if panel.on_key(key): needs_rerun = True; continue   # key eaten by text edit
        if key in (ord('q'), 27):   return False
        if key == ord('r'):         return 'reset'
        if key in (13, ord(' ')):   return True   # start playback

        # Rerun if any detection slider changed
        new_conf     = params[0]['value'] / 100.0
        new_min_area = params[1]['value']
        new_max_area = params[2]['value']
        new_nms      = params[3]['value'] / 100.0
        if (new_conf != confidence or new_min_area != min_area or
                new_max_area != max_area or new_nms != nms_iou):
            needs_rerun = True


# ─────────────────────────────────────────────────────────────────────────────
# Detection loop (live camera + recorded video)
# ─────────────────────────────────────────────────────────────────────────────
def detection_loop(processor, prompts, source_type, source_val,
                   params, tracker, WIN, panel):
    if source_type == 'camera':
        cap = cv2.VideoCapture(source_val)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        mode_label = f"LIVE CAMERA {source_val}"
        is_live = True
    else:
        cap = cv2.VideoCapture(source_val)
        mode_label = os.path.basename(source_val)
        is_live = False

    if not cap.isOpened():
        print("ERROR: Could not open video source."); return 'quit'
    ret, first = cap.read()
    if not ret:
        print("ERROR: Could not read first frame."); return 'quit'
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vh, vw = first.shape[:2]
    scale = min(1.0, 1100/vw)
    dw, dh = int(vw*scale), int(vh*scale)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 1
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    panel.total_frames = max(1, total_frames)
    panel.fps = video_fps; panel.width = dw

    # ── Parameter tuning on first frame ──────────────────────────────────────
    result = param_tuning_screen(processor, prompts, first, params, panel, WIN)
    if result is False:   cap.release(); return 'quit'
    if result == 'reset': cap.release(); return 'reset'
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind after tuning

    cv2.resizeWindow(WIN, dw, dh + SliderPanel.PANEL_H)

    class Relay:
        def __init__(self,p,vh): self.p=p; self.vh=vh
        def cb(self,e,x,y,f,_):
            if y-self.vh>=0: self.p.on_mouse(e,x,y-self.vh)
    cv2.setMouseCallback(WIN, Relay(panel, dh).cb)

    paused = False
    last_frame = first.copy()
    last_masks = np.zeros((0,),dtype=bool)
    last_boxes = np.zeros((0,4)); last_scores = np.zeros((0,))
    last_tracked = {}
    frame_count = 0; save_count = 0
    fps_val = 0.0; fps_timer = time.time()

    while True:
        if panel.seek_frame is not None and not is_live:
            cap.set(cv2.CAP_PROP_POS_FRAMES, panel.seek_frame)
            panel.current_frame = panel.seek_frame
            panel.seek_frame = None; paused = True

        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); panel.current_frame = 0; continue
                break
            last_frame = frame.copy(); frame_count += 1
            if not is_live: panel.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            frame = last_frame.copy()

        confidence = params[0]['value'] / 100.0
        min_area   = params[1]['value']
        max_area   = params[2]['value']
        nms_iou    = params[3]['value'] / 100.0
        every_n    = params[4]['value']
        tracker.max_disappeared = params[5]['value']
        tracker.max_distance    = params[6]['value']

        if (not paused) and (frame_count % every_n == 0):
            # Downscale to 640px wide before SAM3 — the processor upscales internally
            # but a smaller PIL image means faster pre-processing and GPU transfer
            sam_w = min(vw, 640)
            sam_h = int(vh * sam_w / vw)
            small_bgr = cv2.resize(frame, (sam_w, sam_h))
            pil_img = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
            # Scale detected boxes back to original frame coords
            bx_scale = vw / sam_w;  by_scale = vh / sam_h
            last_masks, last_boxes, last_scores = run_sam3_dual(
                processor, prompts, pil_img, confidence, min_area, max_area, nms_iou)
            if len(last_boxes):
                last_boxes = last_boxes * np.array([bx_scale, by_scale, bx_scale, by_scale])
                # Scale masks back too
                last_masks = np.stack([
                    cv2.resize(m.astype(np.uint8), (vw, vh),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    for m in last_masks
                ])
            centroids = [((b[0]+b[2])/2,(b[1]+b[3])/2) for b in last_boxes]
            last_tracked = dict(tracker.update(centroids))
            if frame_count % 10 == 0:
                now = time.time()
                fps_val = 10/max(1e-3, now-fps_timer); fps_timer = now

        display = cv2.resize(frame, (dw, dh))
        sx, sy = dw/vw, dh/vh
        if len(last_masks) > 0:
            overlay_detections(display, last_masks, last_boxes, last_scores,
                               last_tracked, sx, sy)
        draw_hud(display, len(last_masks), tracker.total_seen,
                 fps_val, paused, confidence, mode_label)

        combined = np.vstack([display, panel.render(show_scrubber=not is_live)])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if panel.on_key(key): continue          # key eaten by text edit
        if key in (ord('q'), 27): cap.release(); return 'quit'
        if key == ord('r'):       cap.release(); return 'reset'
        if key == ord('s'):
            fname = f"crystal_{save_count:04d}.png"
            cv2.imwrite(fname, combined); save_count += 1; print(f"Saved {fname}")
        if key == ord(' '):
            paused = not paused; print("Paused." if paused else "Resumed.")

    cap.release(); return 'quit'


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint, device):
    bpe = str(importlib_files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
    ckpt = checkpoint if (checkpoint and os.path.exists(checkpoint)) else \
           (DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else None)
    if ckpt:
        print(f"Loading SAM3: {ckpt}")
        return build_sam3_image_model(bpe_path=bpe, checkpoint_path=ckpt,
                                      load_from_HF=False, device=str(device), eval_mode=True)
    print("Downloading SAM3 from HuggingFace...")
    return build_sam3_image_model(bpe_path=bpe, load_from_HF=True,
                                  device=str(device), eval_mode=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if getattr(torch.backends,"mps",None) and
              torch.backends.mps.is_available() else torch.device("cpu"))
    print(f"Device: {device}  CUDA: {torch.cuda.is_available()}")

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type=="cuda" else nullcontext())

    model = load_model(args.checkpoint, device)

    params = [
        {'name':'Confidence',     'min':5,   'max':95,    'value':65,    'step':1,
         'desc':'SAM3 score threshold (x0.01). Raise to reduce false positives.'},
        {'name':'Min mask area',  'min':50,  'max':50000, 'value':200,   'step':50,
         'desc':'Ignore detections SMALLER than this pixel count.'},
        {'name':'Max mask area',  'min':500, 'max':500000,'value':30000, 'step':500,
         'desc':'Ignore detections LARGER than this — removes whole-frame false positives.'},
        {'name':'NMS IoU thresh', 'min':5,   'max':95,    'value':40,    'step':5,
         'desc':'Overlap threshold (x0.01) to merge duplicate boxes. Lower = stricter.'},
        {'name':'Process every N','min':1,   'max':60,    'value':6,     'step':1,
         'desc':'Run SAM3 every N frames. Higher = faster. Try 10-20 for real-time.'},
        {'name':'Memory (frames)','min':5,   'max':400,   'value':120,   'step':5,
         'desc':'How long tracker remembers a disappeared crystal.'},
        {'name':'Match radius',   'min':10,  'max':1000,  'value':350,   'step':10,
         'desc':'Max pixels a crystal can move and still match. Increase for fast crystals.'},
    ]

    panel   = SliderPanel(1100, params)
    tracker = CentroidTracker()
    WIN = "Crystal Detector — SAM3"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    with autocast_ctx:
        while True:
            # ── Pick reference images via file dialog (multi-select) ──────────
            if args.refs:
                ref_paths = args.refs
            else:
                print("Select reference image(s) — hold Ctrl/Shift to select multiple...")
                ref_paths = pick_images(
                    "Select reference images (Ctrl+click for multiple) — then draw a box on each")
                if not ref_paths: break

            # ── Draw box on each reference ────────────────────────────────────
            confidence = params[0]['value'] / 100.0
            processor  = Sam3Processor(model, confidence_threshold=confidence)
            prompts = []
            cancelled = False
            for idx, ref in enumerate(ref_paths):
                label = f"{idx+1}/{len(ref_paths)}"
                pil_ref, box_ref = setup_screen(ref, label)
                if pil_ref is None:
                    cancelled = True; break
                print(f"Extracting prompt from reference {label}: {os.path.basename(ref)}")
                p, pm = extract_prompt(processor, pil_ref, box_ref)
                prompts.append((p, pm))

            if cancelled or not prompts: break
            print(f"{len(prompts)} reference prompt(s) ready.")

            # ── Choose source ─────────────────────────────────────────────────
            source = source_select_screen(force_video=args.video,
                                          force_camera=args.camera)
            if source is None: break

            tracker.reset()
            result = detection_loop(processor, prompts, source[0], source[1],
                                    params, tracker, WIN, panel)
            print(f"Session: {tracker.total_seen} unique crystals tracked.")

            if result == 'quit': break
            # 'reset' → loops back to file picker

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crystal detector — SAM3")
    ap.add_argument("--refs",       default=None, nargs="+", help="One or more reference images (skips file picker)")
    ap.add_argument("--video",      default=None, help="Video file (skips source-select screen)")
    ap.add_argument("--camera",     type=int, default=None, help="Camera index (skips source-select screen)")
    ap.add_argument("--checkpoint", default=None, help="Path to sam3.pt")
    run(ap.parse_args())
