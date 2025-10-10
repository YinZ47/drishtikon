#!/usr/bin/env python3
"""
Eye Host (FINAL)
- Keeps your ORIGINAL Arduino protocol & UI exactly the same.
- Plugs in robust face detection (ROI fast pass, rotation fallback, Haar backup).
- Adds optional heart monitor window (press 'h' to toggle) driven by motion state.

Serial protocol to MCU (unchanged):
  MODE HOST / MODE AUTO
  G <x> <y>        (normalized -1..1)
  OPEN <o>         (0..1)
  BLINK
  CENTER
  STATUS / PING / HELP (from MCU)

Windows:
  - "Controls": gamma, eyelid %, deadzone, send interval, img size
  - "Face Tracking (YOLOv8 ...)": camera feed + mode/overlays
  - "Heart Monitor": only if --heart or 'h' toggled

Keys:
  t=TRACK, a=AUTO, m=MANUAL, g=auto-gamma toggle
  x/y axis invert, b=blink, o=open (to %), c=close, q=quit
  arrows control gaze in manual
  h toggle heart window | 0 none | 1 brady | 2 tachy | 3 af

"""

import cv2, time, threading, serial, sys, argparse, math, numpy as np
from statistics import mean
from pathlib import Path
try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

# =====–– Ultralytics (YOLOv8) ––=====
from ultralytics import YOLO

# ================= Defaults / UI =================
DEFAULT_FACE_MODEL     = "yolov8n-face.pt"
DEFAULT_GENERIC_MODEL  = "yolov8n.pt"
PERSON_CLASS_ID        = 0
UPPER_BODY_FRACTION    = 0.65
HAAR_URL               = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_FILE              = "haarcascade_frontalface_default.xml"
HAAR_SCALE_FACTOR      = 1.12
HAAR_MIN_NEIGHBORS     = 5
HAAR_MIN_SIZE_FRAC     = 0.10

FONT        = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Face Tracking (YOLOv8 + Gamma, spike-guard + axis invert + loss hold)"
CTRL_WIN    = "Controls"

GAMMA_X100_DEFAULT = 100
EYELID_OPEN_DEFAULT = 92

# ================= Rate/Delta limits (unchanged) =================
MIN_DELTA_NORM = 0.01
MIN_SEND_INTERVAL = 0.05
Y_MAX_PER_CMD = 0.12
Y_DROP_SPIKE_THRESH = 0.25

FACE_LOST_HOLD_S = 0.35
REACQUIRE_STABILIZE_S = 0.12

# ================= One Euro filter (unchanged) =================
class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, dcutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = max(1e-6, t - self.t_prev)
        dx = (x - self.x_prev) / dt
        a_d = self.alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# ================= App State =================
class State:
    def __init__(self):
        self.running = True
        self.mode_tracking = False
        self.mode_manual   = False
        self.serial = None
        # filters
        self.fx = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
        self.fy = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
        # smoothing buffers
        self.recent_x = []
        self.recent_y = []
        # command timing
        self.last_send = 0.0
        self.last_cmd_x = 0.0
        self.last_cmd_y = 0.0
        # gamma
        self.auto_gamma = False
        self.gamma = 1.0
        # manual gaze
        self.manual_x = 0.0
        self.manual_y = 0.0
        # axis inversion
        self.invert_x = True
        self.invert_y = False
        # detection continuity
        self.was_detected_last = False
        self.lost_at = 0.0
        self.reacquired_at = 0.0
        # NEW adaptive detection state
        self.frame_idx = 0
        self.last_box = None       # (x1,y1,x2,y2)
        self.miss_streak = 0
        # FPS EMA (stable display)
        self.prev_t = time.time()
        self.fps_ema = None
        self.fps_alpha = 0.25

        # Heart monitor toggles
        self.show_heart = False
        self.disease = "none"

# ------------------ Serial utils (unchanged behavior) -------------------
def list_available_ports():
    if not list_ports:
        print("pyserial tools not available. Install with: pip install pyserial")
        return []
    return list(list_ports.comports())

def auto_select_port(preferred=None, debug=False):
    ports = list_available_ports()
    if not ports:
        print("[ERROR] No serial ports detected.")
        return None
    if preferred:
        for p in ports:
            if preferred.lower() in (p.description or "").lower() or preferred.lower() in p.device.lower():
                if debug: print(f"[DEBUG] Preferred port match: {p.device} ({p.description})")
                return p.device
    if len(ports) == 1:
        if debug: print(f"[DEBUG] Using sole port: {ports[0].device}")
        return ports[0].device
    for p in ports:
        desc = (p.description or "").lower()
        if "usb" in desc or "arduino" in desc or "wch" in desc or "ch340" in desc:
            if debug: print(f"[DEBUG] Heuristic port: {p.device}")
            return p.device
    print("[INFO] Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} : {p.description}")
    try:
        idx = int(input("Select index: ").strip())
        if 0 <= idx < len(ports):
            return ports[idx].device
    except:
        pass
    print("[ERROR] No selection made.")
    return None

def serial_reader_thread(state: State, debug=False):
    buf = b""
    while state.running and state.serial and state.serial.is_open:
        try:
            data = state.serial.read(128)
            if data:
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    txt = line.decode(errors="ignore").strip()
                    if debug and txt:
                        print(f"[MCU] {txt}")
        except serial.SerialException as e:
            print(f"[ERROR] Serial read: {e}")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected serial error: {e}")
            break

def send_cmd(state: State, cmd, debug=False):
    if state.serial and state.serial.is_open:
        try:
            state.serial.write((cmd.strip()+"\n").encode())
            if debug:
                print(f"[-> MCU] {cmd}")
        except Exception as e:
            print(f"[WARN] Failed sending '{cmd}': {e}")

# ------------------ Modes (unchanged) -------------------
def enter_track(state: State, debug=False):
    if not state.mode_tracking:
        send_cmd(state, "MODE HOST", debug)
        state.mode_tracking = True
        state.mode_manual = False
        state.recent_x.clear()
        state.recent_y.clear()
        state.fx = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
        state.fy = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
        state.miss_streak = 0

def enter_auto(state: State, debug=False):
    if state.mode_tracking or state.mode_manual:
        send_cmd(state, "MODE AUTO", debug)
        state.mode_tracking = False
        state.mode_manual = False

def enter_manual(state: State, debug=False):
    if not state.mode_manual:
        send_cmd(state, "MODE HOST", debug)
        state.mode_manual = True
        state.mode_tracking = False

# ------------------ Gamma / UI helpers (unchanged) -------------------
def apply_gamma(img_bgr, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return img_bgr
    invG = 1.0 / max(1e-6, gamma)
    lut = np.array([((i / 255.0) ** invG) * 255.0 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, lut)

def estimate_mean_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def auto_gamma_update(state: State, frame_bgr, target_mean=125.0, gain=0.08):
    current = estimate_mean_gray(frame_bgr)
    if current <= 1:
        return
    error = (target_mean - current) / 255.0
    mult = math.exp(gain * error)
    state.gamma = float(np.clip(state.gamma * mult, 0.2, 3.0))

# ------------------ Model loading / fallback -------------------
def ensure_haar(debug=False):
    if Path(HAAR_FILE).exists():
        return True
    import urllib.request
    try:
        print("[INFO] Downloading Haar cascade...")
        urllib.request.urlretrieve(HAAR_URL, HAAR_FILE)
        return True
    except Exception as e:
        print(f"[WARN] Haar download failed: {e}")
        return False

def load_models(face_model, generic_model, fallback=True, debug=False):
    face_path = Path(face_model)
    if face_path.exists():
        print(f"[INFO] Using face model: {face_model}")
        return YOLO(face_model), None, True
    if not fallback:
        print(f"[FATAL] Face model '{face_model}' missing, fallback disabled.")
        sys.exit(1)
    print(f"[WARN] Face model missing. Falling back to generic + Haar.")
    generic = YOLO(generic_model)
    haar = None
    if ensure_haar(debug=debug):
        haar = cv2.CascadeClassifier(HAAR_FILE)
        if haar.empty():
            print("[ERROR] Haar cascade load failed.")
            haar = None
    return generic, haar, False

def pick_face_face_model(result, conf, area_priority):
    if result is None or result.boxes is None or len(result.boxes)==0:
        return (None,)*6
    boxes = result.boxes
    best = None
    best_metric = -1
    for i in range(len(boxes)):
        c = float(boxes.conf[i])
        if c < conf: continue
        x1,y1,x2,y2 = boxes.xyxy[i].tolist()
        area = (x2-x1)*(y2-y1)
        metric = area if area_priority else c
        if metric > best_metric:
            best_metric = metric
            best = (x1,y1,x2,y2,c)
    if not best:
        return (None,)*6
    x1,y1,x2,y2,c = best
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    h,w = result.orig_shape[:2]
    return int(cx), int(cy), w, h, (int(x1),int(y1),int(x2),int(y2)), c

def detect_face_fallback(result, haar, conf, area_priority):
    if result is None or result.boxes is None or len(result.boxes)==0 or haar is None:
        return (None,)*6
    h,w = result.orig_shape[:2]
    best_face = None
    best_metric = -1
    for i in range(len(result.boxes)):
        cls = int(result.boxes.cls[i].item())
        if cls != PERSON_CLASS_ID: continue
        pc = float(result.boxes.conf[i].item())
        if pc < conf*0.6: continue
        x1,y1,x2,y2 = [int(v) for v in result.boxes.xyxy[i].tolist()]
        if x2<=x1 or y2<=y1: continue
        upper_h = int((y2-y1)*UPPER_BODY_FRACTION)
        roi = result.orig_img[y1:y1+upper_h, x1:x2]
        if roi.size==0: continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(
            gray, scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=(0,int(upper_h*HAAR_MIN_SIZE_FRAC))
        )
        for (fx,fy,fw2,fh2) in faces:
            area = fw2*fh2
            metric = area if area_priority else pc
            if metric > best_metric:
                ax1 = x1+fx; ay1 = y1+fy; ax2 = ax1+fw2; ay2 = ay1+fh2
                cx = (ax1+ax2)/2.0; cy = (ay1+ay2)/2.0
                best_metric = metric
                best_face = (int(cx),int(cy),w,h,(ax1,ay1,ax2,ay2),pc)
    return best_face if best_face else (None,)*6

# ---- Adaptive detection helpers ----
ROI_EXPAND = 1.55
ROI_REFRESH_INTERVAL = 18
ANGLES_BASE = [-60,-40,-25,25,40,60]  # escalating fallback

def rotate_image_and_matrix(image, angle_deg):
    h,w=image.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2), angle_deg,1.0)
    rotated=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR)
    M_full=np.vstack([M,[0,0,1]])
    M_inv=np.linalg.inv(M_full)
    return rotated,M_full,M_inv

def map_box_back(box,M_inv):
    x1,y1,x2,y2=box
    pts=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]],dtype=float).T
    orig=M_inv @ pts
    xs=orig[0]/orig[2]; ys=orig[1]/orig[2]
    return int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())

def clip_box(box,w,h):
    x1,y1,x2,y2=box
    x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1))
    x2=max(0,min(w-1,x2)); y2=max(0,min(h-1,y2))
    if x2<=x1 or y2<=y1: return None
    return (x1,y1,x2,y2)

def detect_face_adaptive(frame, model, have_face_model, haar, conf, area_priority,
                         last_box, miss_streak, frame_idx, img_size):
    h_full,w_full=frame.shape[:2]

    # ROI fast pass
    if last_box and miss_streak==0 and frame_idx % ROI_REFRESH_INTERVAL != 0:
        x1,y1,x2,y2=last_box
        bw=x2-x1; bh=y2-y1
        cx=(x1+x2)//2; cy=(y1+y2)//2
        rw=int(bw*ROI_EXPAND); rh=int(bh*ROI_EXPAND)
        rx1=max(0,cx-rw//2); ry1=max(0,cy-rh//2)
        rx2=min(w_full,rx1+rw); ry2=min(h_full,ry1+rh)
        sub=frame[ry1:ry2,rx1:rx2]
        sub_size=min(img_size,max(sub.shape[:2]))
        r=model(sub, imgsz=sub_size, verbose=False)[0]
        if have_face_model:
            cx0,cy0,_,_,b0,cf0=pick_face_face_model(r,conf,area_priority)
        else:
            cx0,cy0,_,_,b0,cf0=detect_face_fallback(r,haar,conf,area_priority)
        if cx0 is not None:
            bx1,by1,bx2,by2=b0
            return int(cx0+rx1),int(cy0+ry1),w_full,h_full,(bx1+rx1,by1+ry1,bx2+rx1,by2+ry1),cf0,0.0

    # Upright full frame
    res=model(frame, imgsz=img_size, verbose=False)[0]
    if have_face_model:
        cx,cy,W,H,box,cf=pick_face_face_model(res,conf,area_priority)
    else:
        cx,cy,W,H,box,cf=detect_face_fallback(res,haar,conf,area_priority)
    if cx is not None:
        return cx,cy,W,H,box,cf,0.0

    # Rotation fallback escalates with miss streak
    if miss_streak == 0:
        return (None,)*7
    angles = [-25,25] if miss_streak<3 else ANGLES_BASE
    for ang in angles:
        rot,M,M_inv=rotate_image_and_matrix(frame, ang)
        rres=model(rot, imgsz=img_size, verbose=False)[0]
        if have_face_model:
            rx,ry,_,_,rbox,rconf=pick_face_face_model(rres,conf,area_priority)
        else:
            rx,ry,_,_,rbox,rconf=detect_face_fallback(rres,haar,conf,area_priority)
        if rx is None: continue
        obox=map_box_back(rbox,M_inv)
        obox=clip_box(obox,w_full,h_full)
        if obox is None: continue
        ocx=(obox[0]+obox[2])//2
        ocy=(obox[1]+obox[3])//2
        return ocx,ocy,w_full,h_full,obox,rconf,ang
    return (None,)*7

# ------------------ Draw helpers -------------------
def annotate(display, box, cx, cy, conf, face_mode, gamma_val, mean_gray, mode_text, invx, invy, fps):
    x1,y1,x2,y2 = box
    cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.circle(display,(cx,cy),5,(0,255,0),2)
    label = f"Face {conf*100:.1f}%" if face_mode else f"Det {conf*100:.1f}%"
    cv2.putText(display,label,(x1,max(20,y1-8)),FONT,0.5,(0,255,0),2)
    cv2.putText(display, f"{mode_text} | FPS:{fps:.1f}", (10,25), FONT, 0.65, (0,255,0), 2)
    cv2.putText(display, f"Gamma:{gamma_val:.2f} Mean:{int(mean_gray)} InvX:{'Y' if invx else 'N'} InvY:{'Y' if invy else 'N'}", (10,50), FONT, 0.55, (200,255,0), 2)
    cv2.putText(display, "Keys: t/a/m g autoGamma x/y invert b blink o open c close arrows manual h heart q quit", (10,75), FONT, 0.48, (180,200,255), 1)

def temporal(seq, val, window):
    if window <= 1:
        return val
    seq.append(val)
    if len(seq) > window:
        seq.pop(0)
    return int(mean(seq))

# ------------------ Controls window -------------------
def init_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Gamma x100", CTRL_WIN, GAMMA_X100_DEFAULT, 300, lambda v: None)
    cv2.createTrackbar("Eyelid Open %", CTRL_WIN, EYELID_OPEN_DEFAULT, 100, lambda v: None)
    cv2.createTrackbar("Deadzone %", CTRL_WIN, 5, 30, lambda v: None)
    cv2.createTrackbar("Send ms", CTRL_WIN, 70, 300, lambda v: None)
    cv2.createTrackbar("Img Size", CTRL_WIN, 640, 1280, lambda v: None)

def get_ctrl_values(state: State, args):
    gx100 = cv2.getTrackbarPos("Gamma x100", CTRL_WIN)
    eyelid_open_pct = cv2.getTrackbarPos("Eyelid Open %", CTRL_WIN)
    deadzone_pct = cv2.getTrackbarPos("Deadzone %", CTRL_WIN)
    send_ms = cv2.getTrackbarPos("Send ms", CTRL_WIN)
    img_size = cv2.getTrackbarPos("Img Size", CTRL_WIN)
    gamma = max(20, gx100) / 100.0
    state.gamma = gamma if not state.auto_gamma else state.gamma
    send_interval = max(MIN_SEND_INTERVAL*1000, send_ms) / 1000.0
    img_size = max(160, img_size)
    return gamma, eyelid_open_pct/100.0, deadzone_pct/100.0, send_interval, img_size

# ================== (Optional) Heart monitor pieces ==================
# Motion → velocity state (uses stricter thresholds, dwell) and ECG
class VelocityTracker:
    def __init__(self, alpha=0.18, warmup_sec=2.0,
                 k1=3.8, k2=14.0,
                 dwell_move_on=0.25, dwell_move_off=0.15,
                 dwell_fast_on=0.45, dwell_fast_off=0.30):
        self.alpha=alpha; self.warmup_sec=warmup_sec
        self.k1=k1; self.k2=k2
        self.dwell_move_on=dwell_move_on; self.dwell_move_off=dwell_move_off
        self.dwell_fast_on=dwell_fast_on; self.dwell_fast_off=dwell_fast_off
        self.last=None; self.last_t=None; self.ema=None
        self.warmup_end=None; self.buf=[]; self.baseline=0.0; self.noise=1e-3
        self.state="Stable"; self.hyst_eps=0.0
        self._above_move=None; self._above_fast=None; self._below_move=None; self._below_fast=None
    def _rob(self):
        if not self.buf: return 0.0,1e-3
        arr=np.array(self.buf,dtype=np.float32)
        med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
        sigma=max(1e-4,1.4826*mad); return med,sigma
    def update(self, cx_norm, cy_norm, tnow):
        if self.warmup_end is None: self.warmup_end=tnow+self.warmup_sec
        raw=0.0
        if self.last is not None and self.last_t is not None:
            dt=max(1e-3, tnow-self.last_t)
            dx=cx_norm-self.last[0]; dy=cy_norm-self.last[1]
            raw=(dx*dx+dy*dy)**0.5 / dt
        self.ema = raw if self.ema is None else (self.alpha*raw + (1-self.alpha)*self.ema)
        ready = tnow>=self.warmup_end
        if not ready:
            self.buf.append(self.ema)
            self.baseline,self.noise=self._rob(); self.hyst_eps=0.25*self.noise
        else:
            stable_up=self.baseline+(self.k1-0.5)*self.noise
            if self.ema<stable_up:
                beta=0.01
                self.baseline=(1-beta)*self.baseline+beta*self.ema
                self.noise=(1-beta)*self.noise+beta*abs(self.ema-self.baseline)
                self.hyst_eps=0.25*self.noise
        thr_m=self.baseline+self.k1*self.noise
        thr_f=self.baseline+self.k2*self.noise
        now=tnow
        # dwell timers
        if self.ema>thr_f+self.hyst_eps:
            self._above_fast=now if self._above_fast is None else self._above_fast
            self._above_move=None; self._below_fast=None; self._below_move=None
        elif self.ema>thr_m+self.hyst_eps:
            self._above_move=now if self._above_move is None else self._above_move
            self._above_fast=None; self._below_fast=None; self._below_move=None
        else:
            self._below_move=now if self._below_move is None else self._below_move
            self._below_fast=now if self._below_fast is None else self._below_fast
            self._above_fast=None; self._above_move=None
        # states
        if self.state=="Stable":
            if self._above_fast and (now-self._above_fast)>=self.dwell_fast_on: self.state="Moving fast"; self._above_fast=None
            elif self._above_move and (now-self._above_move)>=self.dwell_move_on: self.state="In motion"; self._above_move=None
        elif self.state=="In motion":
            if self._above_fast and (now-self._above_fast)>=self.dwell_fast_on: self.state="Moving fast"; self._above_fast=None
            elif self._below_move and (now-self._below_move)>=self.dwell_move_off: self.state="Stable"; self._below_move=None
        elif self.state=="Moving fast":
            if self._below_fast and (now-self._below_fast)>=self.dwell_fast_off: self.state="In motion"; self._below_fast=None
        self.last=(cx_norm,cy_norm); self.last_t=tnow
        return raw, self.ema, self.state, ready, self.baseline, self.noise

class HeartRateModel:
    def __init__(self, disease="none"):
        self.set_disease(disease)
        self.hr=float(self.base_bpm); self.scare_bump=0.0
        self.last_t=time.time(); self.stable_accum=0.0; self.cooldown_hold=2.0
    def set_disease(self,disease):
        d=(disease or "none").lower(); self.disease=d
        if d=="bradycardia": self.base_bpm=50; self.noise_std=0.25; self.min_hr=38; self.max_hr=160; self.af_rate=0.0; self.af_amp=(0,0)
        elif d=="tachycardia": self.base_bpm=110; self.noise_std=0.35; self.min_hr=90; self.max_hr=190; self.af_rate=0.0; self.af_amp=(0,0)
        elif d in ("af","afib","fa"): self.base_bpm=95; self.noise_std=0.9; self.min_hr=70; self.max_hr=190; self.af_rate=0.02; self.af_amp=(3,8)
        else: self.base_bpm=72; self.noise_std=0.20; self.min_hr=45; self.max_hr=185; self.af_rate=0.0; self.af_amp=(0,0)
        self.hr=float(self.base_bpm); self.scare_bump=0.0; self.last_t=time.time(); self.stable_accum=0.0
    def update(self, vstate):
        t=time.time(); dt=max(1e-3, t-self.last_t); self.last_t=t
        if vstate=="Stable": self.stable_accum+=dt
        else: self.stable_accum=0.0
        if vstate=="Moving fast": target_bump=25.0; tau=1.1
        elif vstate=="In motion": target_bump=10.0; tau=1.4
        else: target_bump=0.0; tau=2.8
        self.scare_bump += (target_bump - self.scare_bump) * (1.0 - np.exp(-dt/tau))
        self.scare_bump = float(np.clip(self.scare_bump,0.0,25.0))
        target = self.base_bpm + self.scare_bump
        if vstate=="Stable" and self.stable_accum<self.cooldown_hold: target=self.hr
        tau_up=1.0; tau_dn=7.8
        tau = tau_up if target>self.hr else tau_dn
        self.hr += (target - self.hr) * (1.0 - np.exp(-dt/tau))
        self.hr += np.random.normal(0.0, self.noise_std)
        if self.af_rate>0 and np.random.rand()<self.af_rate*dt: self.hr += np.random.uniform(*self.af_amp)
        self.hr=float(np.clip(self.hr, self.min_hr, self.max_hr))
        scared = (vstate=="Moving fast") or (self.scare_bump>8.0)
        return self.hr, scared

class ECGGenerator:
    def __init__(self, disease="none"):
        self.set_disease(disease)
        self.rr=60.0/72.0; self.phase=0.0; self.last_t=time.time()
        self.p_pos=0.12; self.p_w=0.045; self.p_amp=0.08
        self.q_pos=0.24; self.q_w=0.020; self.q_amp=-0.15
        self.r_pos=0.26; self.r_w=0.015; self.r_amp=1.20
        self.s_pos=0.28; self.s_w=0.020; self.s_amp=-0.25
        self.t_pos=0.55; self.t_w=0.10;  self.t_amp=0.35
    def set_disease(self, d):
        d=(d or "none").lower(); self.disease=d
        if d in ("af","afib","fa"): self.rr_jitter=0.11; self.p_scale=0.35; self.extra_amp_jit=0.06
        elif d=="bradycardia": self.rr_jitter=0.02; self.p_scale=1.0; self.extra_amp_jit=0.02
        elif d=="tachycardia": self.rr_jitter=0.03; self.p_scale=0.85; self.extra_amp_jit=0.04
        else: self.rr_jitter=0.02; self.p_scale=1.0; self.extra_amp_jit=0.03
    def _g(self, x, mu, s, a): return a*np.exp(-0.5*((x-mu)/max(1e-6,s))**2)
    def _shape(self, x):
        v  = self._g(x,self.p_pos,self.p_w,self.p_amp*self.p_scale)
        v += self._g(x,self.q_pos,self.q_w,self.q_amp)
        v += self._g(x,self.r_pos,self.r_w,self.r_amp)
        v += self._g(x,self.s_pos,self.s_w,self.s_amp)
        v += self._g(x,self.t_pos,self.t_w,self.t_amp)
        return v
    def step(self, bpm):
        t=time.time(); dt=max(1e-3,t-self.last_t); self.last_t=t
        desired_rr=60.0/max(30.0,min(200.0,bpm))
        self.phase += dt
        while self.phase>=self.rr:
            self.phase-=self.rr
            base_rr=desired_rr; jitter=np.random.normal(0.0,self.rr_jitter)*base_rr
            self.rr=max(0.25, base_rr+jitter)
        x=self.phase/max(1e-6,self.rr)
        v=self._shape(x)*(1.0+np.random.normal(0.0,self.extra_amp_jit))
        return float(v)

class HeartMonitorUI:
    def __init__(self, w=640, h=300):
        self.w,self.h=w,h
        cv2.namedWindow("Heart Monitor", cv2.WINDOW_NORMAL); cv2.resizeWindow("Heart Monitor", w,h)
        self.grid=self._grid(); self.trace=np.zeros_like(self.grid); self.last_y=h//2
        self.scroll_pps=140.0; self.amp=int(0.40*h)
    def _grid(self):
        img=np.zeros((self.h,self.w,3),dtype=np.uint8); img[:]=(30,30,40)
        for y in range(0,self.h,20): cv2.line(img,(0,y),(self.w,y),(45,45,60),1)
        for x in range(0,self.w,40): cv2.line(img,(x,0),(x,self.h),(45,45,60),1)
        return img
    @staticmethod
    def _col(hr):
        pts=[(50,(255,128,0)),(70,(0,255,0)),(100,(0,255,255)),(130,(0,128,255)),(150,(0,0,255))]
        if hr<=pts[0][0]: return pts[0][1]
        if hr>=pts[-1][0]: return pts[-1][1]
        for (h0,c0),(h1,c1) in zip(pts,pts[1:]):
            if h0<=hr<=h1:
                t=(hr-h0)/(h1-h0+1e-9); return tuple(int((1-t)*c0[i]+t*c1[i]) for i in range(3))
        return (0,255,0)
    def render(self, bpm, ecg, fps, scared=False):
        dx=max(1,int(self.scroll_pps/max(1e-3,fps)))
        self.trace=np.roll(self.trace,-dx,axis=1); self.trace[:,-dx:,:]=0
        center=self.h//2; y=int(center-ecg*self.amp); y=max(0,min(self.h-1,y))
        color=self._col(int(round(bpm)))
        for i in range(dx):
            xi=self.w-dx+i
            yi=int(self.last_y+(y-self.last_y)*(i+1)/dx)
            cv2.line(self.trace,(xi,yi),(xi,yi),color,2)
        self.last_y=y
        img=self.grid.copy()
        txt=f"{int(round(bpm))}"; scale=2.6; thick=6
        (tw,th),_=cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx=self.w-tw-20; ty=th+20
        cv2.putText(img,txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+3,cv2.LINE_AA)
        cv2.putText(img,txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)
        cv2.putText(img,"BPM",(tx+tw+12,ty-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,200,200),2,cv2.LINE_AA)
        if scared:
            (bw,bh),_=cv2.getTextSize("SCARED",cv2.FONT_HERSHEY_SIMPLEX,1.0,3)
            cv2.putText(img,"SCARED",(22,32+bh),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,"SCARED",(20,30+bh),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)
        mask=self.trace.astype(bool); img[mask]=self.trace[mask]
        cv2.imshow("Heart Monitor", img)

# ================= Main =================
def build_parser():
    p = argparse.ArgumentParser(description="YOLOv8 Face Tracking -> Arduino HOST gaze control (final, robust).")
    p.add_argument("--port", default="AUTO", help="Serial port (e.g. COM4) or AUTO.")
    p.add_argument("--prefer", help="Preferred port substring when using AUTO.")
    p.add_argument("--list-ports", action="store_true", help="List ports and exit.")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--model", default=DEFAULT_FACE_MODEL)
    p.add_argument("--generic-model", default=DEFAULT_GENERIC_MODEL)
    p.add_argument("--no-fallback", action="store_true")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--area-priority", action="store_true")
    p.add_argument("--temporal-window", type=int, default=3)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    # heart monitor
    p.add_argument("--heart", action="store_true", help="Show heart monitor window at start.")
    p.add_argument("--disease", default="none", choices=["none","bradycardia","tachycardia","af","afib","fa"])
    return p

def main(args):
    state = State()
    state.show_heart = bool(args.heart)
    if args.disease in ("afib","fa"): state.disease="af"
    else: state.disease=args.disease

    if args.list_ports:
        ports = list_available_ports()
        if not ports:
            print("No serial ports found."); return
        print("Available ports:")
        for p in ports: print(f"  {p.device} : {p.description}")
        return

    if args.port.upper() == "AUTO":
        port_to_use = auto_select_port(preferred=args.prefer, debug=args.debug)
        if not port_to_use:
            print("[FATAL] Could not resolve a serial port."); return
    else:
        port_to_use = args.port

    try:
        state.serial = serial.Serial(port_to_use, args.baud, timeout=0.05)
    except serial.SerialException as e:
        print(f"[FATAL] Open {port_to_use} failed: {e}")
        return

    time.sleep(2)
    print(f"[INFO] Connected to {port_to_use}")
    threading.Thread(target=serial_reader_thread, args=(state,args.debug), daemon=True).start()

    model, haar, have_face_model = load_models(
        face_model=args.model,
        generic_model=args.generic_model,
        fallback=not args.no_fallback,
        debug=args.debug
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open camera index {args.camera}")
        return

    init_controls()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # enter AUTO by default (same as your original)
    print("[KEYS] t=tracking | a=autonomous | m=manual | g=auto-gamma | x/y invert | o open | c close | b blink | arrows(manual) | h heart | 0/1/2/3 disease | q quit")
    enter_auto(state, debug=args.debug)
    manual_step = 0.06

    # Heart monitor machinery
    vel = VelocityTracker()
    hr_model = HeartRateModel(disease=state.disease)
    ecg = ECGGenerator(disease=state.disease)
    hr_ui = HeartMonitorUI(640, 300) if state.show_heart else None

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        state.frame_idx += 1

        gamma_slider, eyelid_open, deadzone, send_interval, img_size = get_ctrl_values(state, args)
        mean_gray_pre = estimate_mean_gray(frame)
        if state.auto_gamma:
            auto_gamma_update(state, frame, target_mean=125.0, gain=0.08)
        proc = apply_gamma(frame, state.gamma)
        display = proc.copy()

        # FPS EMA
        now = time.time()
        dt = now - state.prev_t
        state.prev_t = now
        inst_fps = 1.0 / max(1e-3, dt)
        state.fps_ema = inst_fps if state.fps_ema is None else (state.fps_alpha*inst_fps + (1-state.fps_alpha)*state.fps_ema)
        fps_show = state.fps_ema or inst_fps

        if state.mode_manual:
            cv2.putText(display, f"Mode: MANUAL (HOST)", (10,25), FONT, 0.6, (0,200,255), 2)
            cv2.putText(display, f"Manual x:{state.manual_x:+.2f} y:{state.manual_y:+.2f} step:{manual_step:.2f} | FPS:{fps_show:.1f}", (10,50), FONT, 0.55, (0,200,255), 2)
            if now - state.last_send >= send_interval:
                send_cmd(state, f"G {state.manual_x:.3f} {state.manual_y:.3f}", debug=args.debug)
                state.last_send = now
                state.last_cmd_x = state.manual_x
                state.last_cmd_y = state.manual_y

            # keep heart window alive even in manual (no motion input)
            if state.show_heart and hr_ui is not None:
                bpm,_ = hr_model.update("Stable")
                v = ecg.step(bpm)
                hr_ui.render(bpm, v, fps_show, scared=False)

        elif state.mode_tracking:
            # Detect face (adaptive)
            cx, cy, Fw, Fh, box, conf, angle_used = detect_face_adaptive(
                proc, model, have_face_model, haar,
                conf=args.conf,
                area_priority=args.area_priority,
                last_box=state.last_box,
                miss_streak=state.miss_streak,
                frame_idx=state.frame_idx,
                img_size=img_size
            )
            detected = (cx is not None and cy is not None and box is not None)

            if detected:
                state.miss_streak = 0
                if not state.was_detected_last:
                    state.reacquired_at = now
                    state.fx = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
                    state.fy = OneEuro(min_cutoff=1.2, beta=0.03, dcutoff=1.0)
                state.was_detected_last = True
                state.last_box = box

                # small temporal centroid smoothing
                def tavg(seq, v, w):
                    seq.append(v)
                    if len(seq) > w: seq.pop(0)
                    return int(mean(seq))
                cx = tavg(state.recent_x, cx, args.temporal_window)
                cy = tavg(state.recent_y, cy, args.temporal_window)

                annotate(display, box, cx, cy, conf if conf else 0.0,
                         have_face_model, state.gamma, mean_gray_pre,
                         "Mode: TRACK (HOST)", state.invert_x, state.invert_y, fps_show)

                # normalized gaze
                x_norm = (Fw*0.5 - float(cx)) / (Fw*0.5 + 1e-6)
                y_norm = (Fh*0.5 - float(cy)) / (Fh*0.5 + 1e-6)
                if state.invert_x: x_norm = -x_norm
                if state.invert_y: y_norm = -y_norm
                x_norm = float(np.clip(x_norm, -1.0, 1.0))
                y_norm = float(np.clip(y_norm, -1.0, 1.0))

                x_f = state.fx.filter(x_norm, now)
                y_f = state.fy.filter(y_norm, now)

                if abs(x_f) < deadzone: x_f = 0.0
                if abs(y_f) < deadzone: y_f = 0.0

                # spike guard (unchanged)
                dy = y_f - state.last_cmd_y
                if dy < -Y_DROP_SPIKE_THRESH:
                    y_f = state.last_cmd_y - Y_MAX_PER_CMD
                else:
                    if dy > Y_MAX_PER_CMD: y_f = state.last_cmd_y + Y_MAX_PER_CMD
                    elif dy < -Y_MAX_PER_CMD: y_f = state.last_cmd_y - Y_MAX_PER_CMD
                dx = x_f - state.last_cmd_x
                X_MAX_PER_CMD = 0.20
                if dx > X_MAX_PER_CMD: x_f = state.last_cmd_x + X_MAX_PER_CMD
                elif dx < -X_MAX_PER_CMD: x_f = state.last_cmd_x - X_MAX_PER_CMD

                # send at cadence
                if (now - state.reacquired_at) >= REACQUIRE_STABILIZE_S and (now - state.last_send) >= send_interval:
                    if (abs(x_f - state.last_cmd_x) >= MIN_DELTA_NORM) or (abs(y_f - state.last_cmd_y) >= MIN_DELTA_NORM):
                        send_cmd(state, f"G {x_f:.3f} {y_f:.3f}", debug=args.debug)
                        state.last_send = now
                        state.last_cmd_x = x_f
                        state.last_cmd_y = y_f

                # HEART: feed motion from normalized center movement
                if state.show_heart and hr_ui is not None:
                    raw, ema, vstate, ready, base, noise = vel.update(x_norm, y_norm, time.time())
                    bpm, scared = hr_model.update(vstate)
                    v = ecg.step(bpm)
                    hr_ui.render(bpm, v, fps_show, scared=scared)

            else:
                state.miss_streak += 1
                if state.was_detected_last:
                    state.lost_at = now
                state.was_detected_last = False
                cv2.putText(display, f"Mode: TRACK (HOST) | FPS:{fps_show:.1f}", (10,25), FONT, 0.65, (0,0,255), 2)
                cv2.putText(display, "No face (holding pose)", (10,50), FONT, 0.55, (0,0,255), 2)
                # heart gently returns to baseline while lost
                if state.show_heart and hr_ui is not None:
                    bpm, scared = hr_model.update("Stable")
                    v = ecg.step(bpm)
                    hr_ui.render(bpm, v, fps_show, scared=False)

        else:
            # AUTO mode = MCU wanders eye; we just show gamma/fps
            cv2.putText(display, f"Mode: AUTO (board autonomous) | FPS:{fps_show:.1f}", (10,25), FONT, 0.65, (255,255,0), 2)
            cv2.putText(display, f"Gamma:{state.gamma:.2f}", (10,50), FONT, 0.55, (200,255,0), 2)
            if state.show_heart and hr_ui is not None:
                bpm, scared = hr_model.update("Stable")
                v = ecg.step(bpm)
                hr_ui.render(bpm, v, fps_show, scared=False)

        cv2.imshow(WINDOW_NAME, display)
        k = cv2.waitKey(1) & 0xFF

        if k != 255:
            if k == ord('q'):
                state.running = False
            elif k == ord('t'):
                enter_track(state, debug=args.debug)
            elif k == ord('a'):
                enter_auto(state, debug=args.debug)
            elif k == ord('m'):
                enter_manual(state, debug=args.debug)
            elif k == ord('g'):
                state.auto_gamma = not state.auto_gamma
                if not state.auto_gamma:
                    state.gamma = gamma_slider
            elif k == ord('x'):
                state.invert_x = not state.invert_x
            elif k == ord('y'):
                state.invert_y = not state.invert_y
            elif k == ord('b'):
                send_cmd(state, "BLINK", debug=args.debug)
            elif k == ord('o'):
                send_cmd(state, f"OPEN {eyelid_open:.2f}", debug=args.debug)
            elif k == ord('c'):
                send_cmd(state, "OPEN 0", debug=args.debug)
            elif k == ord('h'):
                # toggle heart window
                state.show_heart = not state.show_heart
                if state.show_heart and hr_ui is None:
                    hr_ui = HeartMonitorUI(640, 300)
                if not state.show_heart and hr_ui is not None:
                    cv2.destroyWindow("Heart Monitor"); hr_ui=None
            elif k in (ord('0'),ord('1'),ord('2'),ord('3')):
                dsel = {ord('0'):"none", ord('1'):"bradycardia", ord('2'):"tachycardia", ord('3'):"af"}[k]
                state.disease=dsel; hr_model.set_disease(dsel); ecg.set_disease(dsel)
            elif state.mode_manual:
                if k == 81:   # Left
                    state.manual_x = float(np.clip(state.manual_x + 0.06 * (+1 if not state.invert_x else -1), -1.0, 1.0))
                elif k == 83: # Right
                    state.manual_x = float(np.clip(state.manual_x + 0.06 * (-1 if not state.invert_x else +1), -1.0, 1.0))
                elif k == 82: # Up
                    state.manual_y = float(np.clip(state.manual_y + 0.06 * (+1 if not state.invert_y else -1), -1.0, 1.0))
                elif k == 84: # Down
                    state.manual_y = float(np.clip(state.manual_y + 0.06 * (-1 if not state.invert_y else +1), -1.0, 1.0))

    print("[INFO] Shutting down...")
    try: enter_auto(state, debug=args.debug)
    except: pass
    cap.release()
    cv2.destroyAllWindows()
    if state.serial and state.serial.is_open:
        state.serial.close()
    time.sleep(0.2)

if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
