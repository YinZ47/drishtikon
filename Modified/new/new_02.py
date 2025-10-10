#!/usr/bin/env python3
"""
Eye Host (Stable HUD + Face-Only Anti-Drift)
- Fixes overlays disappearing by drawing a HUD every frame (no silent paths).
- Reuses Haar (no per-call construction). Safer tracker creation (CSRT/MOSSE).
- Face-only: YOLO face if present; else OpenCV DNN face; else Haar.
"""

import cv2, time, threading, serial, sys, argparse, math, numpy as np
from statistics import mean
from pathlib import Path

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------- Config ----------------
DEFAULT_FACE_MODEL     = "yolov8n-face.pt"
DEFAULT_GENERIC_MODEL  = "yolov8n.pt"  # not used for person anymore; we stay face-only

HAAR_URL               = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_FILE              = "haarcascade_frontalface_default.xml"

DNN_PROTO   = "deploy.prototxt"
DNN_MODEL   = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_URL_P   = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_URL_B   = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/55e6d8da089f7c2fbf4a06fe10f6b8f8e9b3a3b6/opencv_face_detector/res10_300x300_ssd_iter_140000.caffemodel"

FONT        = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Face Tracking (stable HUD)"
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
        self.running=True
        self.mode_tracking=False
        self.mode_manual=False
        self.serial=None
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
        # detection continuity
        self.was_detected_last=False; self.reacquired_at=0.0
        self.frame_idx=0
        # FPS
        self.prev_t=time.time(); self.fps_ema=None; self.fps_alpha=0.25
        # detectors
        self.haar=None; self.dnn=None; self.yolo=None; self.have_yolo_face=False
        # tracker
        self.tracker=None
        self.validate_every=3
        self.since_valid=0

# -------------- Serial -------------------
def list_available_ports():
    if not list_ports: return []
    return list(list_ports.comports())

def auto_select_port(preferred=None):
    ports = list_available_ports()
    if not ports: return None
    if preferred:
        for p in ports:
            if preferred.lower() in (p.description or "").lower() or preferred.lower() in p.device.lower():
                return p.device
    # heuristic
    for p in ports:
        desc=(p.description or "").lower()
        if any(k in desc for k in ["usb","arduino","wch","ch340","silabs","cp210"]):
            return p.device
    return ports[0].device

def serial_reader_thread(ser):
    buf=b""
    while ser and ser.is_open:
        try:
            data=ser.read(128)
            if data:
                buf+=data
                while b"\n" in buf:
                    line,buf=buf.split(b"\n",1)
                    txt=line.decode(errors="ignore").strip()
                    if txt: print("[MCU]", txt)
        except Exception as e:
            print("[Serial error]", e)
            break

def send_cmd(state: State, cmd):
    if state.serial and state.serial.is_open:
        try:
            state.serial.write((cmd.strip()+"\n").encode())
        except Exception as e:
            print("[WARN] send failed:", e)

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
        print("[INFO] Downloading Haar...")
        urllib.request.urlretrieve(HAAR_URL, HAAR_FILE)
        c = cv2.CascadeClassifier(HAAR_FILE)
        return c if not c.empty() else None
    except Exception as e:
        print("[WARN] Haar download failed:", e)
        return None

def ensure_dnn():
    if Path(DNN_PROTO).exists() and Path(DNN_MODEL).exists():
        try: return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        except Exception: pass
    import urllib.request
    try:
        if not Path(DNN_PROTO).exists():
            print("[INFO] Downloading DNN prototxt...")
            urllib.request.urlretrieve(DNN_URL_P, DNN_PROTO)
        if not Path(DNN_MODEL).exists():
            print("[INFO] Downloading DNN model...")
            urllib.request.urlretrieve(DNN_URL_B, DNN_MODEL)
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    except Exception as e:
        print("[WARN] DNN download/load failed:", e)
        return None

def load_models(face_model):
    yolo=None; have_yolo=False
    if YOLO is not None and face_model and Path(face_model).exists():
        try:
            yolo = YOLO(face_model)
            have_yolo = True
            print("[INFO] YOLO face ready:", face_model)
        except Exception as e:
            print("[WARN] YOLO face load failed:", e)
    haar = ensure_haar()
    dnn  = ensure_dnn()
    if dnn is not None: print("[INFO] DNN face ready")
    return yolo, have_yolo, haar, dnn

# -------------- Tracker ------------------
def create_tracker():
    # Prefer CSRT (robust), fallback to MOSSE
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
    for i in range(len(r.boxes)):
        c = float(r.boxes.conf[i])
        if c < conf: continue
        x1,y1,x2,y2 = r.boxes.xyxy[i].tolist()
        area=(x2-x1)*(y2-y1)
        m = 0.6*c + 0.4*(area/(frame_bgr.shape[0]*frame_bgr.shape[1]))
        if m>bestm: bestm=m; best=(int(x1),int(y1),int(x2),int(y2),c)
    if not best: return (None,)*6
    x1,y1,x2,y2,cf = best
    cx,cy = (x1+x2)//2,(y1+y2)//2
    h,w = frame_bgr.shape[:2]
    return cx,cy,w,h,(x1,y1,x2,y2),cf

def detect_face_dnn(net, frame_bgr, conf_thr=0.55):
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
                                minSize=(int(0.1*w), int(0.1*h)))
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
    # Prefer DNN check
    if dnn is not None:
        h,w=sub.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(sub,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        dnn.setInput(blob)
        det=dnn.forward()
        for i in range(det.shape[2]):
            if float(det[0,0,i,2])>=0.5: return True
        # fall through to Haar
    if haar is not None:
        g=cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        faces=haar.detectMultiScale(g, scaleFactor=1.08, minNeighbors=5,
                                    minSize=(max(18, sub.shape[1]//6), max(18, sub.shape[0]//6)))
        return len(faces)>0
    return False

# -------------- HUD ----------------------
def draw_hud(display, mode_text, fps, gamma, invx, invy):
    h,w=display.shape[:2]
    # semi-opaque band
    band_h = 64
    overlay = display.copy()
    cv2.rectangle(overlay, (0,0), (w,band_h), (0,0,0), -1)
    display[:] = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
    cv2.putText(display, f"{mode_text} | FPS:{fps:.1f}", (10,24), FONT, 0.65, (0,255,0), 2)
    cv2.putText(display, f"Gamma:{gamma:.2f}  InvX:{'Y' if invx else 'N'} InvY:{'Y' if invy else 'N'}",
                (10,46), FONT, 0.55, (180,255,180), 2)
    cv2.putText(display, "Keys: t/a/m g x y b o c arrows q", (10,64), FONT, 0.48, (180,200,255), 1)

# -------------- Modes --------------------
def enter_track(state: State):
    if not state.mode_tracking:
        send_cmd(state, "MODE HOST")
        state.mode_tracking=True; state.mode_manual=False
        state.recent_x.clear(); state.recent_y.clear()
        state.fx=OneEuro(1.2,0.03,1.0); state.fy=OneEuro(1.2,0.03,1.0)
        state.tracker=None; state.since_valid=0

def enter_auto(state: State):
    if state.mode_tracking or state.mode_manual:
        send_cmd(state, "MODE AUTO")
        state.mode_tracking=False; state.mode_manual=False
        state.tracker=None

def enter_manual(state: State):
    if not state.mode_manual:
        send_cmd(state, "MODE HOST")
        state.mode_manual=True; state.mode_tracking=False
        state.tracker=None

# -------------- CLI ----------------------
def build_parser():
    p=argparse.ArgumentParser(description="Stable HUD face tracking -> Arduino")
    p.add_argument("--port", default="AUTO")
    p.add_argument("--prefer")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--model", default=DEFAULT_FACE_MODEL)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--validate-every", type=int, default=3, help="Validate tracker box as face every N frames")
    return p

# -------------- Main ---------------------
def main(args):
    state=State()
    state.validate_every=max(1, args.validate_every)

    if args.list_ports:
        ports=list_available_ports()
        if not ports: print("No serial ports."); return
        print("Available ports:")
        for p in ports: print(" ", p.device, ":", p.description)
        return

    port = auto_select_port(preferred=args.prefer) if args.port.upper()=="AUTO" else args.port
    if not port:
        print("[FATAL] No serial port"); return
    try:
        state.serial=serial.Serial(port, args.baud, timeout=0.05)
    except Exception as e:
        print("[FATAL] Open port failed:", e); return
    time.sleep(2)
    print("[INFO] Connected:", port)
    threading.Thread(target=serial_reader_thread, args=(state.serial,), daemon=True).start()

    state.yolo, state.have_yolo_face, state.haar, state.dnn = load_models(args.model)

    cap=cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[FATAL] Cannot open camera", args.camera); return

    init_controls()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[KEYS] t=track | a=auto | m=manual | g auto-gamma | x/y invert | o open | c close | b blink | arrows | q quit")
    enter_auto(state)
    manual_step=0.06

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
        proc = apply_gamma(frame, state.gamma if not state.auto_gamma else (auto_gamma_update(state, frame) or state.gamma))

        display = proc.copy()  # ALWAYS draw on 'display'

        if state.mode_manual:
            draw_hud(display, "Mode: MANUAL (HOST)", fps_show, state.gamma, state.invert_x, state.invert_y)
            if now - state.last_send >= send_interval:
                send_cmd(state, f"G {state.manual_x:.3f} {state.manual_y:.3f}")
                state.last_send=now; state.last_cmd_x=state.manual_x; state.last_cmd_y=state.manual_y

        elif state.mode_tracking:
            # 1) Either track (and validate) or detect fresh
            box=None; cx=cy=None; conf=0.0
            used_tracker=False
            if state.tracker is not None:
                ok, tb = state.tracker.update(proc)
                if ok:
                    x,y,w,h=[int(v) for v in tb]
                    cand=(x,y,x+w,y+h)
                    state.since_valid += 1
                    if state.since_valid >= state.validate_every:
                        if face_in_roi(state.haar, state.dnn, proc, cand):
                            state.since_valid = 0
                            box=cand; cx=x+w//2; cy=y+h//2; conf=0.95
                            used_tracker=True
                        else:
                            state.tracker=None; state.since_valid=0
                    else:
                        # use tracker for this frame, validate soon
                        box=cand; cx=x+w//2; cy=y+h//2; conf=0.80
                        used_tracker=True

            if box is None:
                # Fresh detection (prefer YOLO face, else DNN, else Haar)
                if state.yolo is not None and state.have_yolo_face:
                    cx,cy,_,_,box,conf = detect_face_yolo(state.yolo, proc, args.conf, imgsz=img_size)
                if box is None:
                    cx,cy,_,_,box,conf = detect_face_dnn(state.dnn, proc, conf_thr=0.55)
                if box is None:
                    cx,cy,_,_,box,conf = detect_face_haar(state.haar, proc)
                if box is not None:
                    # (Re)build tracker on face box
                    state.tracker = create_tracker()
                    if state.tracker is not None:
                        state.tracker.init(proc, (box[0],box[1],box[2]-box[0],box[3]-box[1]))
                    state.since_valid=0

            # 2) Draw HUD first so it never disappears
            draw_hud(display, "Mode: TRACK (HOST)", fps_show, state.gamma, state.invert_x, state.invert_y)

            # 3) If we have a face, draw and drive
            if box is not None and cx is not None and cy is not None:
                x1,y1,x2,y2=box
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.circle(display,(cx,cy),5,(0,255,0),2)
                cv2.putText(display, f"Face {conf*100:.1f}%{' [TR]' if used_tracker else ''}",
                            (x1,max(20,y1-8)), FONT, 0.5, (0,255,0), 2)

                # temporal smoothing of centroid
                def tavg(seq, v, w=3):
                    seq.append(v); 
                    if len(seq)>w: seq.pop(0)
                    return int(mean(seq))
                cx=tavg(state.recent_x, cx, 3)
                cy=tavg(state.recent_y, cy, 3)

                # normalized gaze
                Fh, Fw = proc.shape[:2]
                x_norm = (Fw*0.5 - float(cx)) / (Fw*0.5 + 1e-6)
                y_norm = (Fh*0.5 - float(cy)) / (Fh*0.5 + 1e-6)
                if state.invert_x: x_norm = -x_norm
                if state.invert_y: y_norm = -y_norm
                x_norm=float(np.clip(x_norm,-1.0,1.0)); y_norm=float(np.clip(y_norm,-1.0,1.0))

                x_f=state.fx.filter(x_norm, now); y_f=state.fy.filter(y_norm, now)
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

                if (now - state.reacquired_at) >= REACQUIRE_STABILIZE_S and (now - state.last_send) >= send_interval:
                    if (abs(x_f-state.last_cmd_x) >= MIN_DELTA_NORM) or (abs(y_f-state.last_cmd_y) >= MIN_DELTA_NORM):
                        send_cmd(state, f"G {x_f:.3f} {y_f:.3f}")
                        state.last_send=now; state.last_cmd_x=x_f; state.last_cmd_y=y_f

                state.was_detected_last=True
            else:
                state.was_detected_last=False  # HUD still shows

        else:
            draw_hud(display, "Mode: AUTO (board autonomous)", fps_show, state.gamma, state.invert_x, state.invert_y)

        # ---- show
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
            elif k==ord('b'): send_cmd(state, "BLINK")
            elif k==ord('o'):
                _, eyelid_open, _, _, _ = get_ctrl_values(state)
                send_cmd(state, f"OPEN {eyelid_open:.2f}")
            elif k==ord('c'): send_cmd(state, "OPEN 0")
            elif state.mode_manual:
                if k==81:   # left
                    state.manual_x=float(np.clip(state.manual_x + 0.06*(+1 if not state.invert_x else -1), -1.0, 1.0))
                elif k==83: # right
                    state.manual_x=float(np.clip(state.manual_x + 0.06*(-1 if not state.invert_x else +1), -1.0, 1.0))
                elif k==82: # up
                    state.manual_y=float(np.clip(state.manual_y + 0.06*(+1 if not state.invert_y else -1), -1.0, 1.0))
                elif k==84: # down
                    state.manual_y=float(np.clip(state.manual_y + 0.06*(-1 if not state.invert_y else +1), -1.0, 1.0))

    # ---- shutdown
    try: enter_auto(state)
    except: pass
    cap.release()
    cv2.destroyAllWindows()
    try:
        if state.serial and state.serial.is_open: state.serial.close()
    except: pass
    time.sleep(0.2)

if __name__ == "__main__":
    args=build_parser().parse_args()
    main(args)
