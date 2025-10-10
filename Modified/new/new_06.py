#!/usr/bin/env python3
"""
Eye Host (Stable HUD + Face-Only Anti-Drift + Heart Sync + Windowed-Speed SCARED/STABILIZING)

- Face-only tracking (YOLO face -> DNN -> Haar) with tracker verification (no chest drift).
- Stable HUD always drawn.
- Heart monitor (ECG + BPM) tied to motion; optional serial sync (HR / HRMODE) to heart MCU.
- SCARED requires sustained fast motion: 75th-percentile speed >= 1.40 norm units/s over 0.50 s.
- IN MOTION when 75th-percentile speed >= 0.70 over 0.25 s.
- STABILIZING shows whenever we're Stable and HR is decaying toward baseline after a SCARED event.

Keys:
  t=TRACK, a=AUTO, m=MANUAL, g=auto-gamma toggle
  x/y axis invert, b=blink, o=open (to %), c=close, q=quit
  arrows in MANUAL
  h toggle heart | 0 none | 1 brady | 2 tachy | 3 af
"""

import cv2, time, threading, serial, sys, argparse, math, numpy as np
from statistics import mean
from pathlib import Path
from collections import deque

# ---------------- Serial helper ----------------
try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

# ---------------- YOLO (optional) -------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------- Defaults --------------------
DEFAULT_FACE_MODEL = "yolov8n-face.pt"

HAAR_URL   = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_FILE  = "haarcascade_frontalface_default.xml"

# OpenCV DNN face model (small, robust)
DNN_PROTO  = "deploy.prototxt"
DNN_MODEL  = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_URL_P  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_URL_B  = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/55e6d8da089f7c2fbf4a06fe10f6b8f8e9b3a3b6/opencv_face_detector/res10_300x300_ssd_iter_140000.caffemodel"

FONT        = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Face Tracking (stable HUD + heart sync)"
CTRL_WIN    = "Controls"

GAMMA_X100_DEFAULT   = 100
EYELID_OPEN_DEFAULT  = 92

MIN_DELTA_NORM       = 0.01
MIN_SEND_INTERVAL    = 0.05
Y_MAX_PER_CMD        = 0.12
Y_DROP_SPIKE_THRESH  = 0.25
REACQUIRE_STABILIZE_S= 0.12

# -------------- Filters -----------------
class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, dcutoff=1.0):
        self.min_cutoff=float(min_cutoff); self.beta=float(beta); self.dcutoff=float(dcutoff)
        self.x_prev=None; self.dx_prev=0.0; self.t_prev=None
    def alpha(self, cutoff, dt):
        r = 2*math.pi*cutoff*dt
        return r/(r+1)
    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev=t; self.x_prev=x; return x
        dt=max(1e-6, t-self.t_prev)
        dx=(x-self.x_prev)/dt
        a_d=self.alpha(self.dcutoff, dt)
        dx_hat=a_d*dx + (1-a_d)*self.dx_prev
        cutoff=self.min_cutoff + self.beta*abs(dx_hat)
        a=self.alpha(cutoff, dt)
        x_hat=a*x + (1-a)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat; self.t_prev=t
        return x_hat

# -------------- State -------------------
class State:
    def __init__(self):
        # app
        self.running=True
        # modes
        self.mode_tracking=False
        self.mode_manual=False
        # serials
        self.serial_eye=None
        self.serial_heart=None
        self.last_hr_sent=0.0
        self.last_hr_val=None
        # filters
        self.fx=OneEuro(1.2,0.03,1.0); self.fy=OneEuro(1.2,0.03,1.0)
        self.recent_x=[]; self.recent_y=[]
        # timing
        self.last_send=0.0; self.last_cmd_x=0.0; self.last_cmd_y=0.0
        # gamma
        self.auto_gamma=False; self.gamma=1.0
        # manual
        self.manual_x=0.0; self.manual_y=0.0
        # axis invert
        self.invert_x=True; self.invert_y=False
        # continuity and perf
        self.was_detected_last=False; self.reacquired_at=0.0
        self.frame_idx=0; self.prev_t=time.time(); self.fps_ema=None; self.fps_alpha=0.25
        # detectors
        self.haar=None; self.dnn=None; self.yolo=None; self.have_yolo_face=False
        # tracker
        self.tracker=None
        self.validate_every=3
        self.since_valid=0
        # heart UI/model
        self.show_heart=False
        self.disease="none"
        # motion debug (off by default)
        self.show_motion_line=False
        self.last_motion_line=""
        # recovery flag (for STABILIZING)
        self.hr_recovering=False

# -------------- Serial utils ------------
def list_available_ports():
    if not list_ports: return []
    return list(list_ports.comports())

def auto_select_port(preferred=None, exclude=None):
    ports=list_available_ports()
    ex=set(exclude or [])
    if not ports: return None
    if preferred:
        for p in ports:
            if p.device in ex: continue
            if preferred.lower() in (p.description or "").lower() or preferred.lower() in p.device.lower():
                return p.device
    # heuristics
    for p in ports:
        if p.device in ex: continue
        desc=(p.description or "").lower()
        if any(k in desc for k in ["usb","arduino","wch","ch340","silabs","cp210"]):
            return p.device
    # fallback
    for p in ports:
        if p.device not in ex: return p.device
    return None

def serial_reader_thread(ser, tag):
    buf=b""
    while ser and ser.is_open:
        try:
            data=ser.read(128)
            if data:
                buf+=data
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    txt=line.decode(errors="ignore").strip()
                    if txt: print(tag, txt)
        except Exception as e:
            break

def send_cmd(ser, cmd):
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip()+"\n").encode())
        except Exception:
            pass

# -------------- Gamma/UI -----------------
def apply_gamma(img_bgr, gamma):
    if abs(gamma-1.0) < 1e-3: return img_bgr
    invG=1.0/max(1e-6,gamma)
    lut=np.array([((i/255.0)**invG)*255.0 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, lut)

def estimate_mean_gray(img_bgr):
    return float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))

def auto_gamma_update(state: State, frame_bgr, target_mean=125.0, gain=0.08):
    current=estimate_mean_gray(frame_bgr)
    if current<=1: return
    error=(target_mean-current)/255.0
    mult=math.exp(gain*error)
    state.gamma=float(np.clip(state.gamma*mult,0.2,3.0))

def init_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Gamma x100", CTRL_WIN, GAMMA_X100_DEFAULT, 300, lambda v: None)
    cv2.createTrackbar("Eyelid Open %", CTRL_WIN, EYELID_OPEN_DEFAULT, 100, lambda v: None)
    cv2.createTrackbar("Deadzone %", CTRL_WIN, 5, 30, lambda v: None)
    cv2.createTrackbar("Send ms", CTRL_WIN, 70, 300, lambda v: None)
    cv2.createTrackbar("Img Size", CTRL_WIN, 512, 1280, lambda v: None)

def get_ctrl_values(state: State):
    gx100=cv2.getTrackbarPos("Gamma x100", CTRL_WIN)
    eyelid_open_pct=cv2.getTrackbarPos("Eyelid Open %", CTRL_WIN)
    deadzone_pct=cv2.getTrackbarPos("Deadzone %", CTRL_WIN)
    send_ms=cv2.getTrackbarPos("Send ms", CTRL_WIN)
    img_size=cv2.getTrackbarPos("Img Size", CTRL_WIN)
    gamma=max(20, gx100)/100.0
    if not state.auto_gamma: state.gamma=gamma
    send_interval = max(MIN_SEND_INTERVAL*1000, send_ms)/1000.0
    img_size=max(160, img_size)
    return gamma, eyelid_open_pct/100.0, deadzone_pct/100.0, send_interval, img_size

# -------------- Detectors ----------------
def ensure_haar():
    if Path(HAAR_FILE).exists():
        c = cv2.CascadeClassifier(HAAR_FILE)
        return c if not c.empty() else None
    import urllib.request
    try:
        urllib.request.urlretrieve(HAAR_URL, HAAR_FILE)
        c = cv2.CascadeClassifier(HAAR_FILE)
        return c if not c.empty() else None
    except Exception:
        return None

def ensure_dnn():
    if Path(DNN_PROTO).exists() and Path(DNN_MODEL).exists():
        try: return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        except Exception: pass
    import urllib.request
    try:
        if not Path(DNN_PROTO).exists():
            urllib.request.urlretrieve(DNN_URL_P, DNN_PROTO)
        if not Path(DNN_MODEL).exists():
            urllib.request.urlretrieve(DNN_URL_B, DNN_MODEL)
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    except Exception:
        return None

def load_models(face_model):
    yolo=None; have_yolo=False
    if YOLO is not None and face_model and Path(face_model).exists():
        try:
            yolo = YOLO(face_model)
            have_yolo = True
        except Exception:
            pass
    haar = ensure_haar()
    dnn  = ensure_dnn()
    return yolo, have_yolo, haar, dnn

# -------------- Tracking -----------------
def create_tracker():
    for maker in [
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerMOSSE_create(),
        lambda: cv2.TrackerMOSSE_create(),
    ]:
        try:
            return maker()
        except Exception:
            continue
    return None

# -------------- Face detect --------------
def detect_face_yolo(model, frame_bgr, conf=0.40, imgsz=512):
    r = model(frame_bgr, imgsz=imgsz, verbose=False)[0]
    if r is None or r.boxes is None or len(r.boxes)==0: return (None,)*6
    best=None; bestm=-1
    area_norm = frame_bgr.shape[0]*frame_bgr.shape[1]
    for i in range(len(r.boxes)):
        c = float(r.boxes.conf[i])
        if c < conf: continue
        x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
        area=(x2-x1)*(y2-y1)
        m = 0.6*c + 0.4*(area/area_norm)
        if m>bestm: bestm=m; best=(int(x1),int(y1),int(x2),int(y2),c)
    if not best: return (None,)*6
    x1,y1,x2,y2,cf = best
    cx,cy = (x1+x2)//2,(y1+y2)//2
    h,w = frame_bgr.shape[:2]
    return cx,cy,w,h,(x1,y1,x2,y2),cf

def detect_face_dnn(net, frame_bgr, conf_thr=0.60):
    if net is None: return (None,)*6
    h,w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    det = net.forward()
    best=None; bestm=-1
    for i in range(det.shape[2]):
        c=float(det[0,0,i,2])
        if c < conf_thr: continue
        x1=int(det[0,0,i,3]*w); y1=int(det[0,0,i,4]*h)
        x2=int(det[0,0,i,5]*w); y2=int(det[0,0,i,6]*h)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
        if x2<=x1 or y2<=y1: continue
        area=(x2-x1)*(y2-y1)
        m=0.6*c + 0.4*(area/(w*h))
        if m>bestm: bestm=m; best=(x1,y1,x2,y2,c)
    if not best: return (None,)*6
    x1,y1,x2,y2,cf = best
    cx,cy=(x1+x2)//2,(y1+y2)//2
    return cx,cy,w,h,(x1,y1,x2,y2),cf

def detect_face_haar(haar, frame_bgr):
    if haar is None: return (None,)*6
    gray=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h,w=gray.shape[:2]
    faces=haar.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                minSize=(int(0.12*w), int(0.12*h)))
    if len(faces)==0: return (None,)*6
    x,y,wf,hf = max(faces, key=lambda r:r[2]*r[3])
    cx,cy=x+wf//2, y+hf//2
    return cx,cy,w,h,(x,y,x+wf,y+hf),1.0

def face_in_roi(haar, dnn, frame_bgr, box):
    x1,y1,x2,y2 = box
    x1=max(0,x1); y1=max(0,y1); x2=min(frame_bgr.shape[1]-1,x2); y2=min(frame_bgr.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return False
    sub = frame_bgr[y1:y2, x1:x2]
    if sub.size==0: return False
    if dnn is not None:
        h,w=sub.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(sub,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        dnn.setInput(blob)
        det=dnn.forward()
        for i in range(det.shape[2]):
            if float(det[0,0,i,2])>=0.60: return True
    if haar is not None:
        g=cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        faces=haar.detectMultiScale(g, scaleFactor=1.08, minNeighbors=6,
                                    minSize=(max(20, sub.shape[1]//5), max(20, sub.shape[0]//5)))
        return len(faces)>0
    return False

# -------------- Heart/motion -----------

class VelocityTracker:
    """
    Windowed-speed classifier to avoid false SCARED from tiny motion.
    - Uses 75th-percentile speed within windows:
        * fast_window=0.50 s -> threshold abs_fast_thr=1.40 (SCARED)
        * move_window=0.25 s -> threshold abs_move_thr=0.70 (IN MOTION)
    - Ignores tiny jitter via pos_epsilon.
    """
    def __init__(self,
                 fast_window=0.50, move_window=0.25,
                 abs_fast_thr=1.40, abs_move_thr=0.70,
                 pos_epsilon=0.02,
                 min_fast_samples=6, min_move_samples=3,
                 warmup_sec=0.6):
        self.fast_window=float(fast_window)
        self.move_window=float(move_window)
        self.abs_fast_thr=float(abs_fast_thr)
        self.abs_move_thr=float(abs_move_thr)
        self.pos_epsilon=float(pos_epsilon)
        self.min_fast_samples=int(min_fast_samples)
        self.min_move_samples=int(min_move_samples)
        self.warmup_sec=float(warmup_sec)

        self.last=None; self.last_t=None
        self.speeds=deque()  # (t, speed)
        self.state="Stable"
        self.warmup_end=None
        # expose for debug
        self.last_raw=0.0
        self.s75_fast=0.0
        self.s75_move=0.0

    def _percentile(self, window):
        tnow = self.speeds[-1][0] if self.speeds else 0.0
        cut = tnow - window
        vals = [s for (t,s) in self.speeds if t >= cut]
        if not vals: return 0.0, 0
        return float(np.percentile(vals, 75)), len(vals)

    def update(self, cx_norm, cy_norm, tnow):
        if self.warmup_end is None: self.warmup_end = tnow + self.warmup_sec

        raw=0.0
        if self.last is not None and self.last_t is not None:
            dt=max(1e-3, tnow-self.last_t)
            dx=cx_norm-self.last[0]; dy=cy_norm-self.last[1]
            if abs(dx)<self.pos_epsilon: dx=0.0
            if abs(dy)<self.pos_epsilon: dy=0.0
            raw=(dx*dx+dy*dy)**0.5 / dt

        self.speeds.append((tnow, raw))
        # trim by fast_window (largest)
        while self.speeds and (tnow - self.speeds[0][0] > max(self.fast_window, self.move_window)):
            self.speeds.popleft()

        # compute robust percentiles in both windows
        self.s75_fast, n_fast = self._percentile(self.fast_window)
        self.s75_move, n_move = self._percentile(self.move_window)

        ready = tnow >= self.warmup_end

        # decide state using windows + sample count
        new_state = "Stable"
        if n_fast >= self.min_fast_samples and self.s75_fast >= self.abs_fast_thr:
            new_state = "Moving fast"
        elif n_move >= self.min_move_samples and self.s75_move >= self.abs_move_thr:
            new_state = "In motion"

        self.state = new_state
        self.last=(cx_norm,cy_norm); self.last_t=tnow; self.last_raw=raw

        # return signature compatible with prior code
        # raw, "ema" (use s75_move), state, ready, base, noise
        return raw, self.s75_move, self.state, ready, self.s75_move, (self.s75_fast - self.s75_move)

class HeartRateModel:
    def __init__(self, disease="none"):
        self.set_disease(disease)
        self.hr=float(self.base_bpm); self.scare_bump=0.0
        self.last_t=time.time(); self.stable_accum=0.0; self.cooldown_hold=1.2
        self.prev_hr=None
    def set_disease(self,disease):
        d=(disease or "none").lower(); self.disease=d
        if d=="bradycardia": self.base_bpm=50; self.noise_std=0.25; self.min_hr=38; self.max_hr=160; self.af_rate=0.0; self.af_amp=(0,0)
        elif d=="tachycardia": self.base_bpm=110; self.noise_std=0.35; self.min_hr=90; self.max_hr=190; self.af_rate=0.0; self.af_amp=(0,0)
        elif d in ("af","afib","fa"): self.base_bpm=95; self.noise_std=0.9; self.min_hr=70; self.max_hr=190; self.af_rate=0.02; self.af_amp=(3,8)
        else: self.base_bpm=72; self.noise_std=0.20; self.min_hr=45; self.max_hr=185; self.af_rate=0.0; self.af_amp=(0,0)
        self.hr=float(self.base_bpm); self.scare_bump=0.0; self.last_t=time.time(); self.stable_accum=0.0; self.prev_hr=None
    def update(self, vstate):
        t=time.time(); dt=max(1e-3, t-self.last_t); self.last_t=t
        if vstate=="Stable": self.stable_accum+=dt
        else: self.stable_accum=0.0
        if vstate=="Moving fast": target_bump=24.0; tau=1.0
        elif vstate=="In motion": target_bump=9.0;  tau=1.3
        else: target_bump=0.0;   tau=2.2
        self.scare_bump += (target_bump - self.scare_bump) * (1.0 - np.exp(-dt/tau))
        self.scare_bump = float(np.clip(self.scare_bump,0.0,26.0))
        target = self.base_bpm + self.scare_bump
        if vstate=="Stable" and self.stable_accum<self.cooldown_hold:
            target=self.hr  # short hold before decay
        tau_up=0.9; tau_dn=5.0  # recovery a bit faster to baseline
        tau = tau_up if target>self.hr else tau_dn
        self.prev_hr = self.hr
        self.hr += (target - self.hr) * (1.0 - np.exp(-dt/tau))
        self.hr += np.random.normal(0.0, self.noise_std)
        if self.af_rate>0 and np.random.rand()<self.af_rate*dt: self.hr += np.random.uniform(*self.af_amp)
        self.hr=float(np.clip(self.hr, self.min_hr, self.max_hr))
        scared = (vstate=="Moving fast")
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
    def render(self, bpm, ecg, fps, status=None):
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
        if status == "SCARED":
            (bw,bh),_=cv2.getTextSize("SCARED",cv2.FONT_HERSHEY_SIMPLEX,1.1,3)
            cv2.putText(img,"SCARED",(22,32+bh),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,"SCARED",(20,30+bh),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),2,cv2.LINE_AA)
        elif status == "STABILIZING":
            (bw,bh),_=cv2.getTextSize("STABILIZING",cv2.FONT_HERSHEY_SIMPLEX,1.0,3)
            cv2.putText(img,"STABILIZING",(22,32+bh),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,"STABILIZING",(20,30+bh),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,0),2,cv2.LINE_AA)
        mask=self.trace.astype(bool); img[mask]=self.trace[mask]
        cv2.imshow("Heart Monitor", img)

# -------------- HUD ----------------------
def draw_hud(display, mode_text, fps, gamma, invx, invy, motion_line=""):
    h,w=display.shape[:2]
    band_h = 80
    overlay = display.copy()
    cv2.rectangle(overlay, (0,0), (w,band_h), (0,0,0), -1)
    display[:] = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
    cv2.putText(display, f"{mode_text} | FPS:{fps:.1f}", (10,24), FONT, 0.65, (0,255,0), 2)
    cv2.putText(display, f"Gamma:{gamma:.2f}  InvX:{'Y' if invx else 'N'} InvY:{'Y' if invy else 'N'}",
                (10,46), FONT, 0.55, (180,255,180), 2)
    cv2.putText(display, "Keys: t/a/m g x y b o c arrows h 0/1/2/3 q", (10,66), FONT, 0.48, (180,200,255), 1)
    if motion_line:
        cv2.putText(display, motion_line, (10,78), FONT, 0.48, (0,255,255), 1, cv2.LINE_AA)

# -------------- Modes --------------------
def enter_track(state: State):
    if not state.mode_tracking:
        send_cmd(state.serial_eye, "MODE HOST")
        state.mode_tracking=True; state.mode_manual=False
        state.recent_x.clear(); state.recent_y.clear()
        state.fx=OneEuro(1.2,0.03,1.0); state.fy=OneEuro(1.2,0.03,1.0)
        state.tracker=None; state.since_valid=0
        state.reacquired_at=time.time()

def enter_auto(state: State):
    if state.mode_tracking or state.mode_manual:
        send_cmd(state.serial_eye, "MODE AUTO")
        state.mode_tracking=False; state.mode_manual=False
        state.tracker=None

def enter_manual(state: State):
    if not state.mode_manual:
        send_cmd(state.serial_eye, "MODE HOST")
        state.mode_manual=True; state.mode_tracking=False
        state.tracker=None

# -------------- CLI ----------------------
def build_parser():
    p=argparse.ArgumentParser(description="Stable HUD face tracking + heart sync -> Arduino")
    # eye
    p.add_argument("--port", default="AUTO")
    p.add_argument("--prefer")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--baud", type=int, default=115200)
    # detection
    p.add_argument("--model", default=DEFAULT_FACE_MODEL)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=3, help="Validate tracker as face every N frames")
    # heart
    p.add_argument("--heart", action="store_true", help="Show heart monitor window at start.")
    p.add_argument("--disease", default="none", choices=["none","bradycardia","tachycardia","af","afib","fa"])
    p.add_argument("--heart-port", default="AUTO", help="Heart serial port (AUTO, explicit, or blank to disable).")
    p.add_argument("--heart-baud", type=int, default=115200)
    p.add_argument("--heart-tx-hz", type=float, default=5.0, help="Transmit HR to heart MCU at this rate.")
    return p

# -------------- Main ---------------------
def main(args):
    state=State()
    state.validate_every=max(1, args.validate_every)
    state.show_heart = bool(args.heart)
    state.disease = "af" if args.disease in ("afib","fa") else args.disease

    if args.list_ports:
        ports=list_available_ports()
        if not ports: print("No serial ports."); return
        print("Available ports:")
        for p in ports: print(" ", p.device, ":", p.description)
        return

    # Eye serial
    port_eye = auto_select_port(preferred=args.prefer) if args.port.upper()=="AUTO" else args.port
    if not port_eye:
        print("[FATAL] No eye serial port"); return
    try:
        state.serial_eye=serial.Serial(port_eye, args.baud, timeout=0.05)
    except Exception as e:
        print("[FATAL] Open eye port failed:", e); return
    time.sleep(2)
    threading.Thread(target=serial_reader_thread, args=(state.serial_eye,"[EYE]"), daemon=True).start()

    # Heart serial (optional)
    heart_enabled=False
    heart_port = None
    if args.heart_port != "":
        if args.heart_port.upper()=="AUTO":
            heart_port = auto_select_port(preferred=None, exclude=[port_eye])
        else:
            heart_port = args.heart_port
        if heart_port:
            try:
                state.serial_heart=serial.Serial(heart_port, args.heart_baud, timeout=0.05)
                time.sleep(1.0)
                heart_enabled=True
                threading.Thread(target=serial_reader_thread, args=(state.serial_heart,"[HEART]"), daemon=True).start()
                send_cmd(state.serial_heart, f"HRMODE {state.disease}")
            except Exception:
                pass

    # Load detectors
    state.yolo, state.have_yolo_face, state.haar, state.dnn = load_models(args.model)

    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera", args.camera); return

    init_controls()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    enter_auto(state)
    manual_step=0.06

    # Heart machinery
    vel = VelocityTracker(
        fast_window=0.50, move_window=0.25,
        abs_fast_thr=1.40, abs_move_thr=0.70,
        pos_epsilon=0.02, min_fast_samples=6, min_move_samples=3,
        warmup_sec=0.6
    )
    hr_model = HeartRateModel(disease=state.disease)
    ecg = ECGGenerator(disease=state.disease)
    hr_ui = HeartMonitorUI(640, 300) if state.show_heart else None

    while state.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue

        state.frame_idx += 1
        now=time.time()
        dt=now - state.prev_t; state.prev_t=now
        fps=1.0/max(1e-3, dt)
        state.fps_ema = fps if state.fps_ema is None else (state.fps_alpha*fps + (1-state.fps_alpha)*state.fps_ema)
        fps_show = state.fps_ema or fps

        gamma_slider, eyelid_open, deadzone, send_interval, img_size = get_ctrl_values(state)
        if state.auto_gamma: auto_gamma_update(state, frame)
        proc = apply_gamma(frame, state.gamma)
        display = proc.copy()

        motion_line=""

        if state.mode_manual:
            draw_hud(display, "Mode: MANUAL (HOST)", fps_show, state.gamma, state.invert_x, state.invert_y)
            if state.show_heart and hr_ui is not None:
                bpm,_ = hr_model.update("Stable")
                v = ecg.step(bpm)
                status = "STABILIZING" if (state.hr_recovering and hr_model.prev_hr is not None and hr_model.prev_hr - bpm > 0.05 and bpm > hr_model.base_bpm + 2.0) else None
                if status is None and bpm <= hr_model.base_bpm + 2.0: state.hr_recovering=False
                hr_ui.render(bpm, v, fps_show, status=status)
                if heart_enabled:
                    maybe_tx_hr(state, bpm, args.heart_tx_hz)
            if now - state.last_send >= send_interval:
                send_cmd(state.serial_eye, f"G {state.manual_x:.3f} {state.manual_y:.3f}")
                state.last_send=now; state.last_cmd_x=state.manual_x; state.last_cmd_y=state.manual_y

        elif state.mode_tracking:
            # tracker update
            box=None; cx=cy=None; conf=0.0; used_tracker=False
            if state.tracker is not None:
                ok, tb = state.tracker.update(proc)
                if ok:
                    x,y,w,h=[int(v) for v in tb]
                    cand=(x,y,x+w,y+h)
                    state.since_valid += 1
                    if state.since_valid >= state.validate_every:
                        if face_in_roi(state.haar, state.dnn, proc, cand):
                            state.since_valid=0
                            box=cand; cx=x+w//2; cy=y+h//2; conf=0.95; used_tracker=True
                        else:
                            state.tracker=None; state.since_valid=0
                    else:
                        box=cand; cx=x+w//2; cy=y+h//2; conf=0.80; used_tracker=True

            # fresh detection if needed
            if box is None:
                if state.yolo is not None and state.have_yolo_face:
                    cx,cy,_,_,box,conf = detect_face_yolo(state.yolo, proc, args.conf, imgsz=img_size)
                if box is None:
                    cx,cy,_,_,box,conf = detect_face_dnn(state.dnn, proc, conf_thr=0.60)
                if box is None:
                    cx,cy,_,_,box,conf = detect_face_haar(state.haar, proc)
                if box is not None:
                    state.tracker = create_tracker()
                    if state.tracker is not None:
                        state.tracker.init(proc, (box[0],box[1],box[2]-box[0],box[3]-box[1]))
                    state.since_valid=0

            draw_hud(display, "Mode: TRACK (HOST)", fps_show, state.gamma, state.invert_x, state.invert_y)

            if box is not None and cx is not None and cy is not None:
                x1,y1,x2,y2=box
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(display,(cx,cy),5,(0,255,0),2)
                cv2.putText(display, f"Face {conf*100:.1f}%{' [TR]' if used_tracker else ''}",
                            (x1,max(20,y1-8)), FONT, 0.5, (0,255,0), 2)

                # temporal centroid smoothing
                def tavg(seq, v, w=3):
                    seq.append(v)
                    if len(seq)>w: seq.pop(0)
                    return int(mean(seq))
                cx=tavg(state.recent_x, cx, 3)
                cy=tavg(state.recent_y, cy, 3)

                # normalized gaze (raw for motion, filtered for servo)
                Fh, Fw = proc.shape[:2]
                x_norm_raw = (Fw*0.5 - float(cx)) / (Fw*0.5 + 1e-6)
                y_norm_raw = (Fh*0.5 - float(cy)) / (Fh*0.5 + 1e-6)
                if state.invert_x: x_norm_raw = -x_norm_raw
                if state.invert_y: y_norm_raw = -y_norm_raw
                x_norm_raw=float(np.clip(x_norm_raw,-1.0,1.0)); y_norm_raw=float(np.clip(y_norm_raw,-1.0,1.0))

                # motion classifier (windowed speeds)
                raw_spd, s75_move, vstate, ready, _, _ = vel.update(x_norm_raw, y_norm_raw, time.time())
                if state.show_motion_line:
                    motion_line=f"raw:{raw_spd:.2f} s75m:{s75_move:.2f} s75f:{vel.s75_fast:.2f} state:{vstate}"
                    draw_hud(display, "Mode: TRACK (HOST)", fps_show, state.gamma, state.invert_x, state.invert_y, motion_line)

                # filtered for servo motion
                x_f=state.fx.filter(x_norm_raw, now); y_f=state.fy.filter(y_norm_raw, now)
                if abs(x_f)<deadzone: x_f=0.0
                if abs(y_f)<deadzone: y_f=0.0

                # spike guard
                dy=y_f-state.last_cmd_y
                if dy < -Y_DROP_SPIKE_THRESH:
                    y_f=state.last_cmd_y - Y_MAX_PER_CMD
                else:
                    if dy>Y_MAX_PER_CMD: y_f=state.last_cmd_y + Y_MAX_PER_CMD
                    elif dy<-Y_MAX_PER_CMD: y_f=state.last_cmd_y - Y_MAX_PER_CMD
                dx=x_f-state.last_cmd_x
                X_MAX_PER_CMD=0.20
                if dx> X_MAX_PER_CMD: x_f=state.last_cmd_x + X_MAX_PER_CMD
                elif dx<-X_MAX_PER_CMD: x_f=state.last_cmd_x - X_MAX_PER_CMD

                # send at cadence
                if (now - state.reacquired_at) >= REACQUIRE_STABILIZE_S and (now - state.last_send) >= send_interval:
                    if (abs(x_f-state.last_cmd_x) >= MIN_DELTA_NORM) or (abs(y_f-state.last_cmd_y) >= MIN_DELTA_NORM):
                        send_cmd(state.serial_eye, f"G {x_f:.3f} {y_f:.3f}")
                        state.last_send=now; state.last_cmd_x=x_f; state.last_cmd_y=y_f

                state.was_detected_last=True

                # HEART: reflect motion + statuses
                bpm, scared = hr_model.update(vstate)
                v = ecg.step(bpm)

                status=None
                if vstate=="Moving fast":
                    status="SCARED"
                    state.hr_recovering=True
                elif vstate=="Stable" and state.hr_recovering:
                    if (hr_model.prev_hr is not None) and (hr_model.prev_hr - bpm > 0.05) and (bpm > hr_model.base_bpm + 2.0):
                        status="STABILIZING"
                    elif bpm <= hr_model.base_bpm + 2.0:
                        state.hr_recovering=False  # recovery complete

                if state.show_heart and hr_ui is not None:
                    hr_ui.render(bpm, v, fps_show, status=status)
                if heart_enabled:
                    maybe_tx_hr(state, bpm, args.heart_tx_hz)
            else:
                state.was_detected_last=False
                bpm, _ = hr_model.update("Stable")
                v = ecg.step(bpm)
                status=None
                if state.hr_recovering:
                    if (hr_model.prev_hr is not None) and (hr_model.prev_hr - bpm > 0.05) and (bpm > hr_model.base_bpm + 2.0):
                        status="STABILIZING"
                    elif bpm <= hr_model.base_bpm + 2.0:
                        state.hr_recovering=False
                if state.show_heart and hr_ui is not None:
                    hr_ui.render(bpm, v, fps_show, status=status)
                if heart_enabled:
                    maybe_tx_hr(state, bpm, args.heart_tx_hz)

        else:
            draw_hud(display, "Mode: AUTO (board autonomous)", fps_show, state.gamma, state.invert_x, state.invert_y)
            bpm, _ = hr_model.update("Stable")
            v = ecg.step(bpm)
            status=None
            if state.hr_recovering:
                if (hr_model.prev_hr is not None) and (hr_model.prev_hr - bpm > 0.05) and (bpm > hr_model.base_bpm + 2.0):
                    status="STABILIZING"
                elif bpm <= hr_model.base_bpm + 2.0:
                    state.hr_recovering=False
            if state.show_heart and hr_ui is not None:
                hr_ui.render(bpm, v, fps_show, status=status)
            maybe_tx_hr(state, bpm, args.heart_tx_hz)

        # UI + keys
        cv2.imshow(WINDOW_NAME, display)
        k=cv2.waitKey(1) & 0xFF
        if k!=255:
            if k==ord('q'): state.running=False
            elif k==ord('t'): enter_track(state)
            elif k==ord('a'): enter_auto(state)
            elif k==ord('m'): enter_manual(state)
            elif k==ord('g'): state.auto_gamma = not state.auto_gamma
            elif k==ord('x'): state.invert_x = not state.invert_x
            elif k==ord('y'): state.invert_y = not state.invert_y
            elif k==ord('b'): send_cmd(state.serial_eye, "BLINK")
            elif k==ord('o'):
                _, eyelid_open, _, _, _ = get_ctrl_values(state)
                send_cmd(state.serial_eye, f"OPEN {eyelid_open:.2f}")
            elif k==ord('c'): send_cmd(state.serial_eye, "OPEN 0")
            elif k==ord('h'):
                state.show_heart = not state.show_heart
                if state.show_heart and hr_ui is None:
                    hr_ui = HeartMonitorUI(640, 300)
                if not state.show_heart and hr_ui is not None:
                    cv2.destroyWindow("Heart Monitor"); hr_ui=None
            elif k in (ord('0'),ord('1'),ord('2'),ord('3')):
                dsel = {ord('0'):"none", ord('1'):"bradycardia", ord('2'):"tachycardia", ord('3'):"af"}[k]
                state.disease=dsel; hr_model.set_disease(dsel); ecg.set_disease(dsel)
                send_cmd(state.serial_heart, f"HRMODE {dsel}")
            elif state.mode_manual:
                if k==81:   # left
                    state.manual_x=float(np.clip(state.manual_x + 0.06*(+1 if not state.invert_x else -1), -1.0, 1.0))
                elif k==83: # right
                    state.manual_x=float(np.clip(state.manual_x + 0.06*(-1 if not state.invert_x else +1), -1.0, 1.0))
                elif k==82: # up
                    state.manual_y=float(np.clip(state.manual_y + 0.06*(+1 if not state.invert_y else -1), -1.0, 1.0))
                elif k==84: # down
                    state.manual_y=float(np.clip(state.manual_y + 0.06*(-1 if not state.invert_y else +1), -1.0, 1.0))

    # shutdown
    try: enter_auto(state)
    except: pass
    cap.release()
    cv2.destroyAllWindows()
    for ser in (state.serial_eye, state.serial_heart):
        try:
            if ser and ser.is_open: ser.close()
        except: pass
    time.sleep(0.2)

def maybe_tx_hr(state: State, bpm_val: float, hz: float):
    if not state.serial_heart: return
    now = time.time()
    min_dt = 1.0 / max(0.5, hz)
    b = int(round(bpm_val))
    if (state.last_hr_val is None) or (abs(b - state.last_hr_val) >= 1) or (now - state.last_hr_sent >= min_dt):
        send_cmd(state.serial_heart, f"HR {b}")
        state.last_hr_sent = now
        state.last_hr_val = b

if __name__ == "__main__":
    args=build_parser().parse_args()
    main(args)
