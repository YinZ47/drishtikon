#!/usr/bin/env python3
# Eye Host (HYBRID, lightweight) – Haar/DNN + Optical Flow + Scheduler
# - Replaces YOLO with your faster stack:
#   * HaarFaceDetector (default) + optional OpenCV DNN Res10 (if files present)
#   * LK optical flow ROI tracker
#   * FrameScheduler: FLOW_ONLY / ROI_DET / FULL / ROT_DET
# - Keeps: Arduino MODE HOST/AUTO, G x y protocol, OneEuro smoothing, spike guards
# - Heart monitor: ECG window + SCARED ramp; toggle with 'h' (auto/on/off)
# - Test without hardware: --port NONE (no serial opens)
#
# Keys:
#   q quit | t TRACK | a AUTO | m MANUAL | arrows (manual) | b blink | o open | c close
#   x/y invert axes | g/G gamma -/+ | 0 none | 1 brady | 2 tachy | 3 af | h heart mode cycle
#   d/D decrease/increase keyframe interval (scheduler)

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2, time, threading, serial, sys, argparse, math, numpy as np
from statistics import mean
from dataclasses import dataclass
from pathlib import Path
try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

# --------------------------- small utils ---------------------------
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def center_of(b): x,y,w,h=b; return (x+0.5*w, y+0.5*h)
def area_of(b): return b[2]*b[3]
def expand_roi(box, s, W, H):
    x,y,w,h=box; cx,cy=x+w/2,y+h/2; nw,nh=w*s,h*s
    nx,ny=cx-nw/2, cy-nh/2
    nx=max(0,min(nx,W-nw)); ny=max(0,min(ny,H-nh))
    return (int(nx),int(ny),int(nw),int(nh))

# --------------------------- One Euro ---------------------------
class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.03, dcutoff=1.0):
        self.min_cutoff=float(min_cutoff); self.beta=float(beta); self.dcutoff=float(dcutoff)
        self.x_prev=None; self.dx_prev=0.0; self.t_prev=None
    def _alpha(self, cutoff, dt):
        r=2*math.pi*cutoff*dt
        return r/(r+1.0)
    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev=t; self.x_prev=x
            return x
        dt=max(1e-6, t-self.t_prev)
        dx=(x-self.x_prev)/dt
        a_d=self._alpha(self.dcutoff, dt)
        dx_hat=a_d*dx + (1-a_d)*self.dx_prev
        cutoff=self.min_cutoff + self.beta*abs(dx_hat)
        a=self._alpha(cutoff, dt)
        x_hat=a*x + (1-a)*self.x_prev
        self.x_prev=x_hat; self.dx_prev=dx_hat; self.t_prev=t
        return x_hat

# --------------------------- Gamma ---------------------------
class GammaLUT:
    def __init__(self, gamma=1.0): self.gamma=None; self.lut=None; self.set_gamma(gamma)
    def set_gamma(self, gamma):
        gamma=clamp(float(gamma),0.2,3.0)
        if self.gamma==gamma and self.lut is not None: return
        self.gamma=gamma; inv=1.0/gamma
        self.lut=np.array([((i/255.0)**inv)*255 for i in range(256)],dtype=np.uint8)
    def apply(self, img): return cv2.LUT(img,self.lut) if self.lut is not None else img

# --------------------------- Camera thread ---------------------------
import queue
class CameraThread:
    def __init__(self, src=0, w=640, h=480, qs=1):
        self.cap=cv2.VideoCapture(src, cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.q=queue.Queue(maxsize=qs); self.running=False
    def start(self): self.running=True; threading.Thread(target=self._loop, daemon=True).start(); return self
    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok: time.sleep(0.01); continue
            if self.q.full():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(f)
    def read(self, timeout=0.5):
        try: return True, self.q.get(timeout=timeout)
        except queue.Empty: return False, None
    def stop(self):
        self.running=False
        try: self.cap.release()
        except Exception: pass

# --------------------------- Detectors (Haar + optional DNN) ---------------------------
class HaarFaceDetector:
    def __init__(self):
        path=os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml")
        self.cc=cv2.CascadeClassifier(path)
        if self.cc.empty(): raise RuntimeError("Failed to load Haar cascade: "+path)
    def detect(self, gray, roi=None):
        x0=y0=0; img=gray
        if roi is not None:
            x0,y0,w,h=roi; img=gray[y0:y0+h, x0:x0+w]
            if img.size==0: return None, 0.0
        dets=self.cc.detectMultiScale(img,1.1,5,minSize=(32,32))
        if len(dets)==0: return None, 0.0
        x,y,w,h=sorted(dets,key=lambda b:b[2]*b[3],reverse=True)[0]
        box=(x+x0,y+y0,w,h)
        conf=float(min(0.99, max(0.60, (w*h)/(gray.shape[0]*gray.shape[1])*10)))
        return box, conf

class DNNFaceDetector:
    """
    Optional OpenCV DNN Res10 SSD (300x300). Put beside script:
      deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
    """
    def __init__(self, proto="deploy.prototxt", weights="res10_300x300_ssd_iter_140000.caffemodel"):
        self.available=False; self.net=None
        if os.path.exists(proto) and os.path.exists(weights):
            try:
                self.net=cv2.dnn.readNetFromCaffe(proto, weights)
                self.available=True
                print("[DNN] Res10 SSD loaded.")
            except Exception as e:
                print("[DNN] Failed to load:", e)
        else:
            print("[DNN] Model files not found; using Haar only.")
    def detect(self, frame_bgr, roi=None, conf_thresh=0.5):
        if not self.available: return None, 0.0
        x0=y0=0; img=frame_bgr
        if roi is not None:
            x0,y0,w,h=roi; img=frame_bgr[y0:y0+h, x0:x0+w]
            if img.size==0: return None, 0.0
        blob=cv2.dnn.blobFromImage(img, 1.0, (300,300), (104,177,123), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections=self.net.forward()
        best=None; bests=0.0
        if detections.ndim!=4 or detections.shape[3]<7: return None, 0.0
        for i in range(detections.shape[2]):
            conf=float(detections[0,0,i,2])
            if conf<conf_thresh: continue
            x1=int(detections[0,0,i,3]*img.shape[1]); y1=int(detections[0,0,i,4]*img.shape[0])
            x2=int(detections[0,0,i,5]*img.shape[1]); y2=int(detections[0,0,i,6]*img.shape[0])
            x1=clamp(x1,0,img.shape[1]-1); y1=clamp(y1,0,img.shape[0]-1)
            x2=clamp(x2,0,img.shape[1]-1); y2=clamp(y2,0,img.shape[0]-1)
            w=max(1,x2-x1); h=max(1,y2-y1)
            if roi is not None: x1+=x0; y1+=y0
            if conf>bests: best=(int(x1),int(y1),int(w),int(h)); bests=conf
        if best is None: return None, 0.0
        return best, bests

# --------------------------- Optical flow tracker ---------------------------
@dataclass
class FlowState:
    prev_gray: np.ndarray=None; prev_pts: np.ndarray=None; roi_box:tuple=None; initialized:bool=False

class FlowTracker:
    def __init__(self, max_corners=80, quality=0.02, min_dist=7):
        self.state=FlowState(); self.max_corners=max_corners; self.quality=quality; self.min_dist=min_dist
    def _init_points(self, gray, box):
        x,y,w,h=[int(v) for v in box]; roi=gray[y:y+h, x:x+w]
        if roi.size==0: return False
        pts=cv2.goodFeaturesToTrack(roi, mask=None, maxCorners=self.max_corners, qualityLevel=self.quality,
                                    minDistance=self.min_dist, blockSize=7, useHarrisDetector=False)
        if pts is None or len(pts)<8: return False
        pts[:,0,0]+=x; pts[:,0,1]+=y
        self.state.prev_pts=pts; self.state.prev_gray=gray; self.state.roi_box=box; self.state.initialized=True; return True
    def reset(self): self.state=FlowState()
    def update(self, gray):
        st=self.state
        if not st.initialized or st.prev_gray is None or st.prev_pts is None or st.roi_box is None:
            return None, 0.0, 0.0
        nxt,stat,err=cv2.calcOpticalFlowPyrLK(st.prev_gray, gray, st.prev_pts, None,
                                              winSize=(21,21), maxLevel=3,
                                              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.03))
        if nxt is None or stat is None: return None, 0.0, 0.0
        old=st.prev_pts[stat.flatten()==1].reshape(-1,2); new=nxt[stat.flatten()==1].reshape(-1,2)
        success=len(new)/max(1,len(st.prev_pts))
        if len(new)<6: return None, success, 0.0
        flow=new-old; dx=float(np.median(flow[:,0])); dy=float(np.median(flow[:,1]))
        x,y,w,h=st.roi_box; new_box=(int(x+dx),int(y+dy),int(w),int(h))
        mag=np.sqrt((flow[:,0]-dx)**2+(flow[:,1]-dy)**2); jitter=float(np.median(mag))
        self.state.prev_pts=new.reshape(-1,1,2); self.state.prev_gray=gray; self.state.roi_box=new_box
        return new_box, success, jitter

# --------------------------- Scheduler ---------------------------
@dataclass
class SchedState:
    frame_idx:int=0; last_detect_idx:int=-999; miss_streak:int=0; last_box:tuple=None; last_conf:float=0.0

class FrameScheduler:
    FLOW_ONLY=0; ROI_DET_SMALL=1; ROI_DET_MED=2; FULL_DET=3; ROT_DET=4
    def __init__(self, k_interval=6, area_thresh_frac=0.015):
        self.k=k_interval; self.area_thresh_frac=area_thresh_frac
    def decide(self, st:SchedState, track_success, jitter, shape):
        H,W=shape[:2]; area_thresh=self.area_thresh_frac*W*H
        conf_term=(1.0-st.last_conf)
        area_term=0.0
        if st.last_box is not None:
            a=area_of(st.last_box); area_term=max(0.0,(area_thresh-a)/max(1e-6,area_thresh))
        jitter_term=min(1.0,jitter/12.0); track_term=(1.0-track_success)
        difficulty=0.45*conf_term + 0.25*area_term + 0.20*jitter_term + 0.10*track_term
        force_key=(st.frame_idx-st.last_detect_idx)>=self.k
        if st.last_box is None or force_key:
            return (FrameScheduler.ROI_DET_SMALL if st.miss_streak==0 else
                    FrameScheduler.ROI_DET_MED   if st.miss_streak==1 else
                    FrameScheduler.FULL_DET), difficulty
        if track_success>=0.5 and difficulty<0.45: return FrameScheduler.FLOW_ONLY, difficulty
        return (FrameScheduler.ROI_DET_SMALL if st.miss_streak==0 else
                FrameScheduler.ROI_DET_MED   if st.miss_streak==1 else
                FrameScheduler.ROT_DET), difficulty

# --------------------------- Velocity tracker (strict) ---------------------------
class VelocityTracker:
    def __init__(self, alpha=0.18, warmup_sec=2.0,
                 k1=3.8, k2=14.0,
                 dwell_move_on=0.25, dwell_move_off=0.15,
                 dwell_fast_on=0.45, dwell_fast_off=0.30):
        self.alpha=alpha; self.warmup_sec=warmup_sec; self.k1=k1; self.k2=k2
        self.dmo=dwell_move_on; self.dmf=dwell_move_off; self.dfo=dwell_fast_on; self.dff=dwell_fast_off
        self.last_c=None; self.last_t=None; self.ema=None
        self.warmup_end=None; self.buf=[]; self.baseline=0.0; self.noise=1e-3
        self.state="Stable"; self.hyst_eps=0.0
        self._am=None; self._af=None; self._bm=None; self._bf=None
    def _rob(self):
        if not self.buf: return 0.0,1e-3
        arr=np.array(self.buf,np.float32); med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med)))
        return med, max(1e-4,1.4826*mad)
    def update(self, box, shape, tnow):
        H,W=shape[:2]; cx,cy=center_of(box)
        if self.warmup_end is None: self.warmup_end=tnow+self.warmup_sec
        raw=0.0
        if self.last_c is not None and self.last_t is not None:
            dt=max(1e-3, tnow-self.last_t)
            dx=(cx-self.last_c[0])/W; dy=(cy-self.last_c[1])/H
            raw=(dx*dx+dy*dy)**0.5 / dt
        self.ema=raw if self.ema is None else (self.alpha*raw + (1-self.alpha)*self.ema)
        ready=tnow>=self.warmup_end
        if not ready:
            self.buf.append(self.ema); self.baseline,self.noise=self._rob(); self.hyst_eps=0.25*self.noise
        else:
            stable_up=self.baseline+(self.k1-0.5)*self.noise
            if self.ema<stable_up:
                beta=0.01
                self.baseline=(1-beta)*self.baseline+beta*self.ema
                self.noise=(1-beta)*self.noise+beta*abs(self.ema-self.baseline)
                self.hyst_eps=0.25*self.noise
        thr_m=self.baseline+self.k1*self.noise; thr_f=self.baseline+self.k2*self.noise
        now=tnow
        if self.ema>thr_f+self.hyst_eps:
            self._af=now if self._af is None else self._af; self._am=self._bm=self._bf=None
        elif self.ema>thr_m+self.hyst_eps:
            self._am=now if self._am is None else self._am; self._af=self._bm=self._bf=None
        else:
            self._bm=now if self._bm is None else self._bm; self._bf=now if self._bf is None else self._bf; self._am=self._af=None
        if self.state=="Stable":
            if self._af and (now-self._af)>=self.dfo: self.state="Moving fast"; self._af=None
            elif self._am and (now-self._am)>=self.dmo: self.state="In motion"; self._am=None
        elif self.state=="In motion":
            if self._af and (now-self._af)>=self.dfo: self.state="Moving fast"; self._af=None
            elif self._bm and (now-self._bm)>=self.dmf: self.state="Stable"; self._bm=None
        else:
            if self._bf and (now-self._bf)>=self.dff: self.state="In motion"; self._bf=None
        self.last_c=(cx,cy); self.last_t=tnow
        return raw,self.ema,self.state,ready,self.baseline,self.noise

# --------------------------- HR + ECG (same calibrated behavior) ---------------------------
class HeartRateModel:
    def __init__(self,disease="none"):
        self.set_disease(disease); self.hr=float(self.base); self.bump=0.0; self.last=time.time(); self.stable=0.0; self.hold=2.0
    def set_disease(self,d):
        d=(d or "none").lower(); self.disease=d
        if d=="bradycardia": self.base=50; self.noise=0.25; self.min=38; self.max=160; self.rate=0.0; self.amp=(0,0)
        elif d=="tachycardia": self.base=110; self.noise=0.35; self.min=90; self.max=190; self.rate=0.0; self.amp=(0,0)
        elif d in ("af","afib","fa"): self.base=95; self.noise=0.9; self.min=70; self.max=190; self.rate=0.02; self.amp=(3,8)
        else: self.base=72; self.noise=0.20; self.min=45; self.max=185; self.rate=0.0; self.amp=(0,0)
        self.hr=float(self.base); self.bump=0.0; self.last=time.time(); self.stable=0.0
    def update(self,vstate):
        t=time.time(); dt=max(1e-3,t-self.last); self.last=t
        if vstate=="Stable": self.stable+=dt
        else: self.stable=0.0
        if vstate=="Moving fast": tgt=25.0; tau=1.1
        elif vstate=="In motion": tgt=10.0; tau=1.4
        else: tgt=0.0; tau=2.8
        self.bump += (tgt-self.bump)*(1.0-np.exp(-dt/tau))
        self.bump=float(np.clip(self.bump,0.0,25.0))
        target=self.base+self.bump
        if vstate=="Stable" and self.stable<self.hold: target=self.hr
        tau_up=1.0; tau_dn=7.8; tau = tau_up if target>self.hr else tau_dn
        self.hr += (target-self.hr)*(1.0-np.exp(-dt/tau))
        self.hr += np.random.normal(0.0,self.noise)
        if self.rate>0 and np.random.rand()<self.rate*dt: self.hr += np.random.uniform(*self.amp)
        self.hr=float(np.clip(self.hr,self.min,self.max))
        scared = (vstate=="Moving fast") or (self.bump>8.0)
        return self.hr, scared

class ECGGenerator:
    def __init__(self,disease="none"):
        self.set_disease(disease); self.rr=60.0/72.0; self.phase=0.0; self.last=time.time()
        self.p=(0.12,0.045,0.08); self.q=(0.24,0.020,-0.15); self.r=(0.26,0.015,1.20); self.s=(0.28,0.020,-0.25); self.t=(0.55,0.10,0.35)
    def set_disease(self,d):
        d=(d or "none").lower(); self.d=d
        if d in ("af","afib","fa"): self.jit=0.11; self.p_scale=0.35; self.amp_jit=0.06
        elif d=="bradycardia": self.jit=0.02; self.p_scale=1.0; self.amp_jit=0.02
        elif d=="tachycardia": self.jit=0.03; self.p_scale=0.85; self.amp_jit=0.04
        else: self.jit=0.02; self.p_scale=1.0; self.amp_jit=0.03
    def _g(self,x,mu,s,a): return a*np.exp(-0.5*((x-mu)/max(1e-6,s))**2)
    def _shape(self,x):
        v  = self._g(x,*self.p[:2], self.p[2]*self.p_scale)
        v += self._g(x,*self.q)
        v += self._g(x,*self.r)
        v += self._g(x,*self.s)
        v += self._g(x,*self.t)
        return v
    def step(self,bpm):
        t=time.time(); dt=max(1e-3,t-self.last); self.last=t
        desired=60.0/max(30.0,min(200.0,bpm)); self.phase+=dt
        while self.phase>=self.rr:
            self.phase-=self.rr
            base=desired; jitter=np.random.normal(0.0,self.jit)*base
            self.rr=max(0.25, base+jitter)
        x=self.phase/max(1e-6,self.rr)
        return float(self._shape(x)*(1.0+np.random.normal(0.0,self.amp_jit)))

class HeartMonitorUI:
    def __init__(self,w=640,h=300):
        self.w,self.h=w,h
        cv2.namedWindow("Heart Monitor",cv2.WINDOW_NORMAL); cv2.resizeWindow("Heart Monitor",w,h)
        self.grid=self._grid(); self.trace=np.zeros_like(self.grid); self.last_y=h//2
        self.scroll_pps=140.0; self.amp=int(0.40*h)
    def _grid(self):
        img=np.zeros((self.h,self.w,3),np.uint8); img[:]=(30,30,40)
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
                t=(hr-h0)/(h1-h0+1e-9)
                return tuple(int((1-t)*c0[i]+t*c1[i]) for i in range(3))
        return (0,255,0)
    def render(self,bpm,ecg,fps,scared=False):
        dx=max(1,int(self.scroll_pps/max(1e-3,fps)))
        self.trace=np.roll(self.trace,-dx,axis=1); self.trace[:,-dx:,:]=0
        y=int(self.h//2 - ecg*self.amp); y=clamp(y,0,self.h-1)
        color=self._col(int(round(bpm)))
        for i in range(dx):
            xi=self.w-dx+i
            yi=int(self.last_y+(y-self.last_y)*(i+1)/dx)
            cv2.line(self.trace,(xi,yi),(xi,yi),color,2)
        self.last_y=y
        img=self.grid.copy()
        txt=f"{int(round(bpm))}"; scale=2.6; thick=6
        (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
        tx=self.w-tw-20; ty=th+20
        cv2.putText(img,txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thick+3,cv2.LINE_AA)
        cv2.putText(img,txt,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)
        cv2.putText(img,"BPM",(tx+tw+12,ty-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(200,200,200),2,cv2.LINE_AA)
        if scared:
            cv2.putText(img,"SCARED",(22,62),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(img,"SCARED",(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2,cv2.LINE_AA)
        mask=self.trace.astype(bool); img[mask]=self.trace[mask]
        cv2.imshow("Heart Monitor", img)

# --------------------------- Serial (optional) ---------------------------
def list_available_ports():
    if not list_ports: return []
    return list(list_ports.comports())

def auto_select_port(preferred=None):
    ports=list_available_ports()
    if not ports: return None
    if preferred:
        for p in ports:
            if preferred.lower() in (p.description or "").lower() or preferred.lower() in p.device.lower():
                return p.device
    if len(ports)==1: return ports[0].device
    for p in ports:
        desc=(p.description or "").lower()
        if "usb" in desc or "arduino" in desc or "wch" in desc or "ch340" in desc:
            return p.device
    return ports[0].device

def serial_reader_thread(ser, debug=False):
    buf=b""
    while ser and ser.is_open:
        try:
            data=ser.read(128)
            if data:
                buf+=data
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    txt=line.decode(errors="ignore").strip()
                    if debug and txt: print(f"[MCU] {txt}")
        except Exception: break

def send_cmd(ser, cmd, debug=False):
    if ser and ser.is_open:
        try:
            ser.write((cmd.strip()+"\n").encode())
            if debug: print(f"[-> MCU] {cmd}")
        except Exception as e:
            if debug: print(f"[WARN] send '{cmd}': {e}")

# --------------------------- App ---------------------------
class EyeHostApp:
    def __init__(self, args):
        cv2.setNumThreads(0)
        self.args=args
        # Serial (optional)
        self.ser=None
        if args.port.upper()!="NONE":
            port = auto_select_port(args.prefer) if args.port.upper()=="AUTO" else args.port
            if port:
                try:
                    self.ser=serial.Serial(port, args.baud, timeout=0.05)
                    time.sleep(2)
                    threading.Thread(target=serial_reader_thread, args=(self.ser,args.debug), daemon=True).start()
                    print(f"[INFO] Connected to {port}")
                except Exception as e:
                    print(f"[WARN] Serial open failed: {e}")
        else:
            print("[INFO] Running with no serial (simulation).")

        # Video & gamma
        self.cam=CameraThread(args.camera, 640, 480).start()
        self.gamma=GammaLUT(1.0)

        # Detectors
        self.haar=HaarFaceDetector()
        self.dnn=DNNFaceDetector()  # optional
        self.flow=FlowTracker()
        self.sched=FrameScheduler(k_interval=args.key_interval)
        self.st=SchedState()

        # Modes
        self.mode_tracking=False; self.mode_manual=False
        self.invert_x=True; self.invert_y=False

        # Filters / send limiting
        self.fx=OneEuro(1.2,0.03,1.0); self.fy=OneEuro(1.2,0.03,1.0)
        self.last_send=0.0; self.last_cmd_x=0.0; self.last_cmd_y=0.0
        self.deadzone=0.05; self.send_interval=0.07
        self.Y_MAX_PER_CMD=0.12; self.Y_DROP_SPIKE_THRESH=0.25
        self.X_MAX_PER_CMD=0.20

        # Heart monitor
        self.vel=VelocityTracker()
        dz=args.disease.lower()
        if dz in ("afib","fa"): dz="af"
        self.hr=HeartRateModel(dz); self.ecg=ECGGenerator(dz)
        self.heart_mode=args.heart_mode  # auto|on|off
        self.show_heart = (self.heart_mode in ("auto","on"))
        self.hr_ui = HeartMonitorUI(640,300) if self.show_heart else None
        self.current_disease=dz

        # FPS EMA
        self.prev_t=time.time(); self.fps_ema=None; self.fps_alpha=0.25

        # ROI/rotate params
        self.roi_expand=1.6; self.rotate_angles=[-15,15]

        # Manual control
        self.manual_x=0.0; self.manual_y=0.0; self.manual_step=0.06

        # Auto toggle thresholds
        self.FPS_HIDE=18.0; self.FPS_SHOW=22.0; self.GRACE=2.0
        self.low_since=None; self.high_since=None

        # Start in AUTO
        self.enter_auto()

    # ---- Modes
    def enter_track(self):
        if not self.mode_tracking:
            send_cmd(self.ser,"MODE HOST",self.args.debug)
            self.mode_tracking=True; self.mode_manual=False
            self.fx=OneEuro(1.2,0.03,1.0); self.fy=OneEuro(1.2,0.03,1.0)
            self.st.miss_streak=0
            print("[MODE] TRACK")
    def enter_auto(self):
        if self.mode_tracking or self.mode_manual:
            send_cmd(self.ser,"MODE AUTO",self.args.debug)
        self.mode_tracking=False; self.mode_manual=False
        print("[MODE] AUTO")
    def enter_manual(self):
        if not self.mode_manual:
            send_cmd(self.ser,"MODE HOST",self.args.debug)
            self.mode_manual=True; self.mode_tracking=False
            print("[MODE] MANUAL")

    # ---- Detection helpers
    def _maybe_rotate(self,bgr,a):
        h,w=bgr.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
        rot=cv2.warpAffine(bgr,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(114,114,114))
        return rot,M
    def _detect_once(self, frame_bgr, gray, roi_local):
        # prefer DNN if present, otherwise Haar
        if self.dnn.available:
            box,conf=self.dnn.detect(frame_bgr, roi=roi_local, conf_thresh=0.5)
            if box is not None: return box,conf
        return self.haar.detect(gray, roi_local)
    def _detect_any(self, frame_bgr, gray, mode, roi_hint):
        roi=None
        if roi_hint is not None:
            H,W=gray.shape[:2]; roi=expand_roi(roi_hint, self.roi_expand, W, H)
        if mode==FrameScheduler.ROT_DET:
            for ang in self.rotate_angles:
                rot,M=self._maybe_rotate(frame_bgr, ang)
                gray_rot=cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
                box,conf=self._detect_once(rot, gray_rot, None)
                if box is not None:
                    x,y,w,h=box
                    pts=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]],dtype=np.float32)
                    Minv=cv2.invertAffineTransform(M); ones=np.ones((4,1),dtype=np.float32)
                    mapped=(np.hstack([pts,ones]) @ Minv.T).astype(np.float32)
                    x1,y1=mapped[:,0].min(), mapped[:,1].min(); x2,y2=mapped[:,0].max(), mapped[:,1].max()
                    return (int(max(0,x1)),int(max(0,y1)),int(x2-x1),int(y2-y1)), conf
            return self._detect_once(frame_bgr, gray, roi)
        elif mode==FrameScheduler.FULL_DET:
            return self._detect_once(frame_bgr, gray, None)
        else:
            return self._detect_once(frame_bgr, gray, roi)

    # ---- Drawing
    def _annotate(self, img, box, vstate, fps, disease):
        x,y,w,h=box
        color = (0,200,0) if vstate=="Stable" else ((0,200,255) if vstate=="In motion" else (0,0,255))
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        header=f"FPS: {fps:.1f} | Disease: {disease}"
        cv2.putText(img, header, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, vstate, (x, y+h+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # ---- Loop
    def run(self):
        print("[KEYS] t TRACK | a AUTO | m MANUAL | g/G gamma -/+ | x/y invert | b BLINK | o OPEN | c close | arrows(manual) | 0/1/2/3 disease | d/D keyframe-interval | h heart | q quit")
        while True:
            ok, frame = self.cam.read(timeout=0.5)
            if not ok or frame is None: continue

            # FPS EMA
            now=time.time(); dt=now-self.prev_t; self.prev_t=now
            inst_fps=1.0/max(1e-3,dt)
            self.fps_ema = inst_fps if self.fps_ema is None else (self.fps_alpha*inst_fps + (1-self.fps_alpha)*self.fps_ema)
            fps_show = self.fps_ema or inst_fps

            # Heart auto-toggle (if auto mode)
            if self.heart_mode=="auto":
                if self.show_heart and fps_show<self.FPS_HIDE:
                    self.low_since = now if self.low_since is None else self.low_since
                    if (now-self.low_since)>=self.GRACE:
                        if self.hr_ui is not None: cv2.destroyWindow("Heart Monitor"); self.hr_ui=None
                        self.show_heart=False; self.low_since=None; self.high_since=None
                else:
                    self.low_since=None
                if (not self.show_heart) and fps_show>self.FPS_SHOW:
                    self.high_since = now if self.high_since is None else self.high_since
                    if (now-self.high_since)>=self.GRACE:
                        self.hr_ui=HeartMonitorUI(640,300); self.show_heart=True; self.high_since=None; self.low_since=None
                else:
                    if self.show_heart: self.high_since=None

            # Apply gamma
            frame=self.gamma.apply(frame)
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Manual mode
            if self.mode_manual:
                vis=frame.copy()
                cv2.putText(vis, f"Mode: MANUAL (HOST) | FPS:{fps_show:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                if (now-self.last_send)>=self.send_interval:
                    send_cmd(self.ser, f"G {self.manual_x:.3f} {self.manual_y:.3f}", self.args.debug)
                    self.last_send=now; self.last_cmd_x=self.manual_x; self.last_cmd_y=self.manual_y
                if self.show_heart and self.hr_ui is not None:
                    bpm,_=self.hr.update("Stable"); v=self.ecg.step(bpm); self.hr_ui.render(bpm,v,fps_show,False)
                cv2.imshow("EyeTracker", vis)
                key=cv2.waitKey(1)&0xFF
                if not self._handle_key(key): break
                continue

            # AUTO (no host gaze)
            if not self.mode_tracking:
                vis=frame.copy()
                cv2.putText(vis, f"Mode: AUTO (board) | FPS:{fps_show:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
                if self.show_heart and self.hr_ui is not None:
                    bpm,_=self.hr.update("Stable"); v=self.ecg.step(bpm); self.hr_ui.render(bpm,v,fps_show,False)
                cv2.imshow("EyeTracker", vis)
                key=cv2.waitKey(1)&0xFF
                if not self._handle_key(key): break
                continue

            # TRACKING mode
            # Flow update first
            track_box, track_success, jitter = self.flow.update(gray)
            if track_box is not None:
                H,W=gray.shape[:2]; x,y,w,h=track_box
                x=int(clamp(x,0,W-1)); y=int(clamp(y,0,H-1))
                w=int(clamp(w,1,W-x)); h=int(clamp(h,1,H-y))
                track_box=(x,y,w,h)

            self.st.frame_idx+=1; self.sched.k=self.args.key_interval
            mode,_ = self.sched.decide(self.st,
                                       track_success if track_box is not None else 0.0,
                                       jitter if track_box is not None else 0.0,
                                       frame.shape)

            if mode==FrameScheduler.FLOW_ONLY and track_box is not None:
                box=track_box; conf=self.st.last_conf
            else:
                hint=self.st.last_box if self.st.last_box is not None else track_box
                box, conf = self._detect_any(frame, gray, mode, hint)
                if box is None:
                    self.st.miss_streak+=1; box=track_box if track_box is not None else None; conf=0.0
                else:
                    self.st.miss_streak=0; self.st.last_detect_idx=self.st.frame_idx
                    self.flow.reset(); self.flow._init_points(gray, box)

            if box is not None: self.st.last_box=box; self.st.last_conf=conf

            # Velocity → HR
            vstate="Stable"
            if box is not None:
                _, ema, vstate, _, _, _ = self.vel.update(box, frame.shape, time.time())
            bpm, scared = self.hr.update(vstate); ecgv=self.ecg.step(bpm)
            if self.show_heart and self.hr_ui is not None:
                self.hr_ui.render(bpm, ecgv, fps_show, scared)

            # Compose and send gaze
            vis=frame.copy()
            if box is not None:
                self._annotate(vis, box, vstate, fps_show, self.current_disease)
                H,W=frame.shape[:2]; cx,cy=center_of(box)
                x_norm=clamp((cx-W/2)/(W/2),-1.0,1.0); y_norm=clamp((cy-H/2)/(H/2),-1.0,1.0)
                if self.invert_x: x_norm=-x_norm
                if self.invert_y: y_norm=-y_norm
                x_f=self.fx.filter(x_norm, now); y_f=self.fy.filter(y_norm, now)
                if abs(x_f)<self.deadzone: x_f=0.0
                if abs(y_f)<self.deadzone: y_f=0.0
                dy=y_f-self.last_cmd_y
                if dy<-self.Y_DROP_SPIKE_THRESH: y_f=self.last_cmd_y - self.Y_MAX_PER_CMD
                else:
                    if dy> self.Y_MAX_PER_CMD: y_f=self.last_cmd_y + self.Y_MAX_PER_CMD
                    elif dy<-self.Y_MAX_PER_CMD: y_f=self.last_cmd_y - self.Y_MAX_PER_CMD
                dx=x_f-self.last_cmd_x
                if dx> self.X_MAX_PER_CMD: x_f=self.last_cmd_x + self.X_MAX_PER_CMD
                elif dx<-self.X_MAX_PER_CMD: x_f=self.last_cmd_x - self.X_MAX_PER_CMD
                if (now-self.last_send)>=self.send_interval:
                    if (abs(x_f-self.last_cmd_x)>=0.01) or (abs(y_f-self.last_cmd_y)>=0.01):
                        send_cmd(self.ser, f"G {x_f:.3f} {y_f:.3f}", self.args.debug)
                        self.last_send=now; self.last_cmd_x=x_f; self.last_cmd_y=y_f
            else:
                cv2.putText(vis, f"Mode: TRACK (HOST) | FPS:{fps_show:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                cv2.putText(vis, "No face (holding pose)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

            cv2.imshow("EyeTracker", vis)
            key=cv2.waitKey(1)&0xFF
            if not self._handle_key(key): break

        # shutdown
        self.cam.stop(); cv2.destroyAllWindows()
        if self.ser and self.ser.is_open: self.ser.close()

    # ---- key handling
    def _handle_key(self, k):
        if k==255: return True
        if k==ord('q'): return False
        elif k==ord('t'): self.enter_track()
        elif k==ord('a'): self.enter_auto()
        elif k==ord('m'): self.enter_manual()
        elif k==ord('g'): self.gamma.set_gamma(self.gamma.gamma-0.1)
        elif k==ord('G'): self.gamma.set_gamma(self.gamma.gamma+0.1)
        elif k==ord('x'): self.invert_x=not self.invert_x
        elif k==ord('y'): self.invert_y=not self.invert_y
        elif k==ord('b'): send_cmd(self.ser,"BLINK",self.args.debug)
        elif k==ord('o'): send_cmd(self.ser,f"OPEN 0.92",self.args.debug)
        elif k==ord('c'): send_cmd(self.ser,"OPEN 0",self.args.debug)
        elif k==ord('0'): self._set_disease("none")
        elif k==ord('1'): self._set_disease("bradycardia")
        elif k==ord('2'): self._set_disease("tachycardia")
        elif k==ord('3'): self._set_disease("af")
        elif k==ord('h'): self._cycle_heart()
        elif self.mode_manual:
            if k==81:   # Left
                self.manual_x = float(np.clip(self.manual_x + self.manual_step * (+1 if not self.invert_x else -1), -1.0, 1.0))
            elif k==83: # Right
                self.manual_x = float(np.clip(self.manual_x + self.manual_step * (-1 if not self.invert_x else +1), -1.0, 1.0))
            elif k==82: # Up
                self.manual_y = float(np.clip(self.manual_y + self.manual_step * (+1 if not self.invert_y else -1), -1.0, 1.0))
            elif k==84: # Down
                self.manual_y = float(np.clip(self.manual_y + self.manual_step * (-1 if not self.invert_y else +1), -1.0, 1.0))
            elif k==ord('['): self.manual_step=max(0.01, self.manual_step-0.01)
            elif k==ord(']'): self.manual_step=min(0.25, self.manual_step+0.01)
        elif k==ord('d'): self.args.key_interval=max(1, self.args.key_interval-1)
        elif k==ord('D'): self.args.key_interval=min(12, self.args.key_interval+1)
        return True

    def _cycle_heart(self):
        if self.heart_mode=="auto":
            self.heart_mode="on"; self.show_heart=True
            if self.hr_ui is None: self.hr_ui=HeartMonitorUI(640,300)
        elif self.heart_mode=="on":
            self.heart_mode="off"; self.show_heart=False
            if self.hr_ui is not None: cv2.destroyWindow("Heart Monitor"); self.hr_ui=None
        else:
            self.heart_mode="auto"; self.show_heart=True
            if self.hr_ui is None: self.hr_ui=HeartMonitorUI(640,300)
        self.low_since=None; self.high_since=None

    def _set_disease(self, name):
        self.current_disease=name
        self.hr.set_disease(name); self.ecg.set_disease(name)
        print(f"[HR] disease set → {name}")

# --------------------------- CLI ---------------------------
def build_parser():
    p=argparse.ArgumentParser(description="Lightweight Eye Host (Haar/DNN + LK flow) with Arduino HOST control and ECG")
    p.add_argument("--port", default="AUTO", help="Serial port (e.g., COM4), AUTO, or NONE to run without hardware")
    p.add_argument("--prefer", help="Preferred substring for AUTO port match")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--key-interval", type=int, default=6, help="Frames between forced detector refreshes")
    p.add_argument("--heart-mode", default="auto", choices=["auto","on","off"])
    p.add_argument("--disease", default="none", choices=["none","bradycardia","tachycardia","af","afib","fa"])
    return p

def main():
    args=build_parser().parse_args()
    app=EyeHostApp(args)
    app.run()

if __name__=="__main__":
    main()
