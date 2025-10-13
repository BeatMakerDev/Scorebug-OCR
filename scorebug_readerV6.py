#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scorebug Reader â€“ Broadcast OCR for Game Clock + Play Clock
===========================================================
Dependencies (install first):
    pip install opencv-python pillow mss pytesseract numpy

What it does
------------
â€¢ Capture from a screen region (MSS) or camera (OpenCV), default canvas 1920Ã—1080.
â€¢ Tk/ttk UI with 16:9 live viewer, perâ€‘ROI (GAME / PLAY) controls in a scrollable panel.
â€¢ Draw two independent ROIs (left-click drag). Toggle target via radio buttons.
â€¢ Multiple OCR pipelines:
    (a) Digital clock pipeline (M:SS) for white/bright glyphs on dark.
    (b) Sevenâ€‘segment pipeline for PLAY clock (two digits).
    (c) Legacy redâ€‘LED method via HSV red masking (+ morphology).
    (d) Template/NCC fallback using generated digit templates.
    (e) Tesseract fallback (digit/colon whitelist, PSM 7).
â€¢ Projectionâ€‘based splitter to separate merged digits.
â€¢ Autoâ€‘calibration sweep that searches parameter combos to match the operatorâ€™s â€œground truthâ€ entry.
â€¢ Reliability: median smoothing, jump/countdown guards, threaded capture+OCR.
â€¢ Atomic TXT/JSON/CSV writers compatible with OBS text source â€œRead from fileâ€.

Quick start
-----------
1) Run:  python scorebug_reader.py
2) Pick â€œScreenâ€ or â€œCameraâ€. For Screen, press â€œPick Regionâ€ to set a capture box.
3) Select GAME or PLAY, then draw an ROI on the viewer. Repeat for the other clock.
4) (Optional) Set Tesseract path (â€¦/tesseract.exe on Windows).
5) Click â€œStartâ€. OCR results appear at top; files write to ./out by default.

Tips
----
â€¢ Press â€œSpaceâ€ to pause/resume preview. Mouse wheel zooms (preview only).
â€¢ Rightâ€‘click clears the active ROI. â€˜aâ€™ triggers autoâ€‘cal for the active ROI.
â€¢ If your play clock is red LEDs, enable â€œLegacy Red LEDâ€ for PLAY.

Test mode
---------
    python scorebug_reader.py --test-image "/path/to/image.png"
You can draw ROIs on the still image and run the same pipelines.

Author: ChatGPT â€“ GPTâ€‘5 Thinking
Direction: James Cromwell
License: MIT
"""
from __future__ import annotations

import os, sys, time, json, math, threading, queue, itertools, tempfile, csv, argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import cv2
from PIL import Image, ImageTk

# Optional MSS for screen capture
try:
    import mss
    HAVE_MSS = True
except Exception:
    HAVE_MSS = False

# Optional Tesseract
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ----------------------------- Utilities ---------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _atomic_replace(tmp: str, path: str, retries: int = 3, delay: float = 0.05):
    """Try to atomically replace tmp -> path. On PermissionError attempt safe fallbacks.

    Retries a few times, then falls back to copying the tmp contents over the target
    (this is not strictly atomic but works when os.replace is blocked by another
    reader like OBS). Errors are written to stderr if all attempts fail.
    """
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # transient lock; wait and retry
            time.sleep(delay)
        except Exception as ex:
            # unexpected error â€” try once to copy as fallback
            try:
                with open(tmp, "rb") as fr, open(path, "wb") as fw:
                    fw.write(fr.read())
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return
            except Exception:
                sys.stderr.write(f"atomic replace error: {ex}\n")
                raise
    # After retries, attempt fallback copy (best-effort)
    try:
        with open(tmp, "rb") as fr, open(path, "wb") as fw:
            fw.write(fr.read())
        try:
            os.remove(tmp)
        except Exception:
            pass
        return
    except PermissionError:
        # Last attempt: try to make target writable and replace
        try:
            if os.path.exists(path):
                try:
                    os.chmod(path, 0o666)
                except Exception:
                    pass
            os.replace(tmp, path)
            return
        except Exception as e:
            sys.stderr.write(f"atomic write ultimately failed: {e}\n")
    except Exception as e:
        sys.stderr.write(f"atomic write fallback failed: {e}\n")

def atomic_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    _atomic_replace(tmp, path)

def atomic_write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    _atomic_replace(tmp, path)

def atomic_write_csv(path: str, rows: List[List[Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    _atomic_replace(tmp, path)

def draw_safe_rect(img, rect, color=(0,255,0), thickness=2):
    (x,y,w,h) = rect
    x = clamp(x, 0, img.shape[1]-1); y = clamp(y, 0, img.shape[0]-1)
    w = clamp(w, 1, img.shape[1]-x); h = clamp(h, 1, img.shape[0]-y)
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness, lineType=cv2.LINE_AA)

def median_smooth(history: List[str], k: int = 3) -> str:
    if not history:
        return ""
    tail = history[-k:]
    # Pick most frequent in the tail
    best = ""
    bestc = 0
    for s in set(tail):
        c = tail.count(s)
        if c > bestc:
            best, bestc = s, c
    return best

# ----------------------------- Capture ------------------------------------

class CaptureSource:
    def __init__(self, mode="screen", camera_index=0, screen_box=None, desired_size=(1920,1080)):
        self.mode = mode
        self.camera_index = camera_index
        self.screen_box = screen_box  # dict for mss
        self.desired_size = desired_size
        self.cap = None
        self.mss = mss.mss() if (mode=="screen" and HAVE_MSS) else None

    def open(self):
        # Release existing handle if present
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.mode == "camera":
            # Try several common OpenCV backends; None means let OpenCV choose.
            try_backends = [None]
            for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_VFW", "CAP_FFMPEG"):
                if hasattr(cv2, name):
                    try_backends.append(getattr(cv2, name))
            for backend in try_backends:
                try:
                    cap = cv2.VideoCapture(self.camera_index) if backend is None else cv2.VideoCapture(self.camera_index, backend)
                    if not cap or not cap.isOpened():
                        try:
                            cap.release()
                        except Exception:
                            pass
                        continue
                    # Try to set resolution (may be ignored by some backends)
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_size[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_size[1])
                    except Exception:
                        pass
                    self.cap = cap
                    return True
                except Exception:
                    # try next backend
                    continue
            # all backends failed
            return False

        elif self.mode == "screen":
            if not HAVE_MSS:
                raise RuntimeError("mss not installed. pip install mss")
            if self.screen_box is None:
                # Fullscreen primary
                mon = self.mss.monitors[1]
                self.screen_box = {"top": mon["top"], "left": mon["left"], "width": mon["width"], "height": mon["height"]}
            return True

        else:
            return False

    def read(self):
        if self.mode == "camera":
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame
        elif self.mode == "screen":
            sct_img = self.mss.grab(self.screen_box)
            img = np.array(Image.frombytes("RGB", sct_img.size, sct_img.rgb))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return np.array(img)
        else:
            return None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

# ------------------------- Preprocessing & OCR ----------------------------

@dataclass
class ROIConfig:
    enabled: bool = True
    legacy_red_led: bool = False
    gamma: float = 1.0
    contrast: float = 1.0  # multiply after gamma
    brightness: int = 0    # add after contrast
    thr_mode: str = "adaptive"  # adaptive|otsu|fixed
    thr: int = 140
    adaptive_block: int = 31
    adaptive_C: int = 7
    morph_open: int = 2  # default increased to remove small noise clusters
    morph_close: int = 1
    use_tesseract: bool = True
    template_fallback: bool = True
    expected_len: int = 4  # GAME "M:SS" is 4 or 5 incl colon; PLAY is 2
    kind: str = "GAME"     # or "PLAY"
    # Debloom processing for PLAY: when enabled, apply an additional
    # morphological operation to counteract bloom/halation that can cause
    # adjacent LED segments to merge. This is especially helpful when the
    # play clock is small or far away and the camera optics blur the red
    # segments into a continuous bar. The feature is toggled via a
    # checkbutton in the play clock tab of the UI and persists in saved
    # profiles.
    debloom: bool = False

@dataclass
class ROIState:
    rect: Optional[Tuple[int,int,int,int]] = None
    history: List[str] = field(default_factory=list)
    last_value: str = ""

DIGIT_TEMPLATES = {}

def ensure_digit_templates(font_scale=1.6, thickness=2):
    """Create simple 28x48 grayscale templates 0-9 using cv2.putText (no external files)."""
    global DIGIT_TEMPLATES
    if DIGIT_TEMPLATES:
        return DIGIT_TEMPLATES
    for d in range(10):
        canvas = np.zeros((48, 28), np.uint8)
        cv2.putText(canvas, str(d), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        DIGIT_TEMPLATES[d] = canvas
    return DIGIT_TEMPLATES

def apply_preprocess(img: np.ndarray, cfg: ROIConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gray, bin)"""
    # gamma
    g = max(0.05, float(cfg.gamma))
    lut = np.array([((i/255.0) ** (1.0/g)) * 255.0 for i in range(256)], dtype=np.uint8)
    adj = cv2.LUT(img, lut)
    # contrast/brightness
    adj = cv2.convertScaleAbs(adj, alpha=float(cfg.contrast), beta=int(cfg.brightness))
    gray = cv2.cvtColor(adj, cv2.COLOR_BGR2GRAY)
    # Apply a small median filter to reduce saltâ€‘andâ€‘pepper noise.  A 3Ã—3 kernel
    # removes isolated bright/dark pixels while preserving segment edges.  This
    # helps the OCR remain robust on noisy broadcast feeds without adding much
    # latency.  We perform this before sharpening so that the unsharp mask
    # operates on a slightly denoised image.
    gray = cv2.medianBlur(gray, 3)
    # slight sharpen
    gray = cv2.GaussianBlur(gray, (0,0), 0.8)
    unsharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0,0), 2.0), -0.5, 0)
    gray = unsharp
    # threshold
    if cfg.legacy_red_led:
        hsv = cv2.cvtColor(adj, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 120, 80]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 80]); upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        binimg = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((cfg.morph_close, cfg.morph_close), np.uint8)) if cfg.morph_close>1 else mask
        if cfg.morph_open>1:
            binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((cfg.morph_open, cfg.morph_open), np.uint8))
        return gray, binimg
    if cfg.thr_mode == "adaptive":
        block = int(cfg.adaptive_block) if int(cfg.adaptive_block)%2==1 else int(cfg.adaptive_block)+1
        binimg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, int(cfg.adaptive_C))
    elif cfg.thr_mode == "otsu":
        _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, binimg = cv2.threshold(gray, int(cfg.thr), 255, cv2.THRESH_BINARY)
    if cfg.morph_open>1:
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((cfg.morph_open, cfg.morph_open), np.uint8))
    if cfg.morph_close>1:
        binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, np.ones((cfg.morph_close, cfg.morph_close), np.uint8))
    return gray, binimg

def split_two_digits(binimg: np.ndarray) -> List[np.ndarray]:
    """Projection-based splitter. Returns up to two digit crops (left->right)."""
    h, w = binimg.shape[:2]
    colsum = np.sum(binimg>0, axis=0).astype(np.float32)
    # Find valley between two peaks
    # Smooth a bit
    colsum = cv2.GaussianBlur(colsum.reshape(1,-1), (1,0), 3).ravel()
    # threshold to only columns with some ink
    active = np.where(colsum > 0.05*np.max(colsum))[0]
    if len(active)==0:
        return []
    x0, x1 = active[0], active[-1]
    mid = (x0+x1)//2
    # search small band around mid for minimum
    band =  int(0.15*w)
    lo = clamp(mid-band, 0, w-1); hi = clamp(mid+band, 0, w-1)
    valley = int(lo + np.argmin(colsum[lo:hi+1]))
    # produce two masks
    left = binimg[:, :valley]
    right = binimg[:, valley:]
    return [left, right]

def seven_seg_read(binimg: np.ndarray) -> str:
    """Read a two-digit seven-seg number by sampling 7 segment regions per digit."""
    parts = split_two_digits(binimg)
    if len(parts)==0:
        parts = [binimg]  # maybe already one digit (fallback)
    out = ""
    for part in parts[:2]:
        h, w = part.shape[:2]
        if w < 6 or h < 10:
            continue
        # Normalize
        tgtw = 28; tgth = 48
        norm = cv2.resize(part, (tgtw, tgth), interpolation=cv2.INTER_AREA)
        # Segment sample positions relative to template (seven segments a,b,c,d,e,f,g)
        seg = {}
        seg['a'] = np.mean(norm[3:7, 6:22]) > 40
        seg['b'] = np.mean(norm[8:22, 21:25]) > 40
        seg['c'] = np.mean(norm[26:40, 21:25]) > 40
        seg['d'] = np.mean(norm[41:45, 6:22]) > 40
        seg['e'] = np.mean(norm[26:40, 3:7]) > 40
        seg['f'] = np.mean(norm[8:22, 3:7]) > 40
        seg['g'] = np.mean(norm[23:27, 6:22]) > 40
        pattern = tuple(int(seg[k]) for k in ('a','b','c','d','e','f','g'))
        # Map pattern to digit
        mapping = {
            (1,1,1,1,1,1,0): '0',
            (0,1,1,0,0,0,0): '1',
            (1,1,0,1,1,0,1): '2',
            (1,1,1,1,0,0,1): '3',
            (0,1,1,0,0,1,1): '4',
            (1,0,1,1,0,1,1): '5',
            (1,0,1,1,1,1,1): '6',
            (1,1,1,0,0,0,0): '7',
            (1,1,1,1,1,1,1): '8',
            (1,1,1,1,0,1,1): '9',
        }
        out += mapping.get(pattern, '?')
    return out

def template_ncc_fallback(binimg: np.ndarray) -> str:
    """Try to read up to 4 chars by template matching 0-9 and ':' built from digits."""
    tmpls = ensure_digit_templates()
    # try to split by connected components into up to 4 boxes
    cnts, _ = cv2.findContours((binimg>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 20]
    boxes = sorted(boxes, key=lambda b: b[0])[:4]
    out = ""
    for (x,y,w,h) in boxes:
        crop = cv2.resize(binimg[y:y+h, x:x+w], (28,48), interpolation=cv2.INTER_AREA)
        best, bestv = None, -1
        for d, tpl in tmpls.items():
            res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
            v = float(res.max())
            if v > bestv:
                bestv, best = v, str(d)
        # crude colon detection: very thin tall blob
        if h > 0.7*binimg.shape[0] and w < 0.25*binimg.shape[1]:
            out += ":"
        else:
            out += best if best is not None else "?"
    return out

def tesseract_read(gray_or_bin: np.ndarray, whitelist="0123456789:", psm=7) -> str:
    if not HAVE_TESS:
        return ""
    cfg = f"-l eng --oem 1 --psm {int(psm)} -c tessedit_char_whitelist={whitelist}"
    if len(gray_or_bin.shape)==2:
        im = gray_or_bin
    else:
        im = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    # Upscale to help OCR
    im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(im, config=cfg)
    return txt.strip().replace(" ", "")

def read_game_clock(gray: np.ndarray, binimg: np.ndarray, cfg: ROIConfig) -> str:
    # Prefer white-on-black -> invert if needed to make digits white
    inv = binimg
    # Heuristic: if background mostly white, invert
    if np.mean(inv) > 127:
        inv = 255 - inv
    # Try tesseract for M:SS
    if cfg.use_tesseract and HAVE_TESS:
        s = tesseract_read(inv, whitelist="0123456789:", psm=7)
        # Basic cleanup
        s = s.replace("::", ":").replace("â€”", "-")
        # normalize MM:SS or M:SS
        if len(s)>=3 and ":" in s:
            parts = s.split(":")
            if len(parts)==2 and all(p.isdigit() for p in parts):
                m, sec = parts
                if len(sec)==1: sec = "0"+sec
                return f"{int(m)}:{sec[:2]}"
    # Fallback to template/NCC
    if cfg.template_fallback:
        s = template_ncc_fallback(inv)
        s = s.replace("::", ":")
        if ":" in s:
            chunks = [c for c in s if c.isdigit() or c==":"]
            s = "".join(chunks)
            # Try to coerce to M:SS
            if ":" in s:
                p = s.split(":")
                if len(p)>=2 and all(x.isdigit() for x in p[:2]):
                    m = str(int(p[0]))
                    sec = p[1].ljust(2,"0")[:2]
                    return f"{m}:{sec}"
    return ""

def read_play_clock(gray: np.ndarray, binimg: np.ndarray, cfg: ROIConfig) -> str:
    """Read the play clock using a robust sevenâ€‘segment pipeline with fallbacks.

    The play clock is typically displayed as a twoâ€‘digit sevenâ€‘segment LED/LCD. We
    perform morphological cleanâ€‘up and optional inversion to normalise the
    binary mask before applying the sevenâ€‘segment reader. If that fails, we
    fall back to Tesseract OCR and template matching.
    """
    # For legacy red LED displays, the mask is already the red channel threshold.
    if cfg.legacy_red_led:
        clean = binimg.copy()
    else:
        # Create a working copy to avoid mutating caller's binimg
        clean = binimg.copy()
        # Close small gaps within segments and remove noise: closing followed by opening
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
        # Invert if the background is mostly white so digits become white on black
        if np.mean(clean) > 127:
            clean = 255 - clean
    # Apply optional deblooming step: additional morphological open with a
    # larger kernel to separate merged segments caused by optical bloom when
    # the play clock is distant or blurred.  This runs only when the
    # `debloom` flag is set in the play configuration.  The 5Ã—5 kernel
    # reduces thick bridges between segments without erasing the digits.
    if cfg.debloom:
        debloom_kernel = np.ones((5, 5), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, debloom_kernel, iterations=1)
    # Primary attempt: sevenâ€‘segment decode
    s = seven_seg_read(clean)
    if len(s) == 2 and s.isdigit():
        return s
    # Fallback: Tesseract OCR restricted to digits
    if cfg.use_tesseract and HAVE_TESS:
        s2 = tesseract_read(clean, whitelist="0123456789", psm=7)
        digits = ''.join([c for c in s2 if c.isdigit()])
        if len(digits) >= 1:
            return digits[:2].rjust(2, "0")
    # Final fallback: template matching
    if cfg.template_fallback:
        s3 = template_ncc_fallback(clean)
        digits_only = ''.join([c for c in s3 if c.isdigit()])
        if len(digits_only) >= 2:
            return digits_only[:2]
    return ""

# ---------------------------- Auto Calibration ----------------------------

def score_candidate(text: str, expected: str, expected_len: int) -> float:
    if not text:
        return -10.0
    # Penalize wrong length heavily
    if len(text) != len(expected):
        return -5.0 * abs(len(text)-len(expected))
    # Reward matches by character
    score = 0.0
    for a, b in zip(text, expected):
        score += 1.5 if a==b else -1.0
    return score

def auto_calibrate(sample_bgr: np.ndarray, cfg: ROIConfig, kind: str, expected: str) -> ROIConfig:
    """Gridâ€‘search a few parameter combos to fit expected text quickly."""
    best_cfg = None
    best_score = -1e9
    # Parameter ranges (kept small for speed)
    gammas = [0.7, 1.0, 1.3, 1.6]
    contrasts = [0.9, 1.0, 1.2]
    brights = [-20, 0, 20]
    thr_modes = ["adaptive","otsu","fixed"]
    thr_vals = [110, 130, 150, 170]
    for g, a, b in itertools.product(gammas, contrasts, brights):
        cfg.gamma, cfg.contrast, cfg.brightness = g, a, b
        for tm in thr_modes:
            cfg.thr_mode = tm
            if tm=="fixed":
                thiter = thr_vals
            else:
                thiter = [cfg.thr]
            for th in thiter:
                cfg.thr = th
                gray, binimg = apply_preprocess(sample_bgr, cfg)
                if kind=="GAME":
                    text = read_game_clock(gray, binimg, cfg)
                else:
                    text = read_play_clock(gray, binimg, cfg)
                s = score_candidate(text, expected, cfg.expected_len)
                if s>best_score:
                    best_score = s
                    best_cfg = ROIConfig(**vars(cfg))
    return best_cfg if best_cfg else cfg

# ----------------------------- GUI / App ----------------------------------

class App:
    def __init__(self, root, args):
        self.root = root
        root.title("Scorebug Reader â€“ Game/Play Clock OCR")
        self.args = args

        self.capture_mode = tk.StringVar(value="screen" if HAVE_MSS else "camera")
        self.camera_index = tk.IntVar(value=0)
        self.screen_box = None
        self.paused = False
        self.tess_path = tk.StringVar(value=pytesseract.pytesseract.tesseract_cmd if HAVE_TESS else "")

        self.game_cfg = ROIConfig(kind="GAME", expected_len=4)
        self.play_cfg = ROIConfig(kind="PLAY", expected_len=2, legacy_red_led=False)

        self.game_state = ROIState()
        self.play_state = ROIState()

        self.current_target = tk.StringVar(value="GAME")
        self.out_dir = tk.StringVar(value=os.path.abspath("./out"))

        # Output format (txt, json, csv).  The selected format determines
        # how the program writes out the current game and play clock values.
        # Each value is written to its own file (game.<fmt> and play.<fmt>).
        self.out_format = tk.StringVar(value="txt")

        # Threading
        self.frame_q = queue.Queue(maxsize=2)
        self.stop_flag = threading.Event()

        self._canvas_img_id = None   # single canvas image item id
        self._tk_im = None          # keep PhotoImage reference
        self._build_ui()
        # Threads will be started when the user chooses to initialise the input.
        self.threads_started = False
        # Track last capture settings to detect changes and reinitialise capture
        self._capture_last_mode = None
        self._capture_last_index = None
        self._capture_last_box = None

    # ---------- UI ----------
    def _build_ui(self):
        root = self.root
        root.geometry("1200x720")
        root.configure(bg="#111")

        top = ttk.Frame(root); top.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Label(top, text="Source:").pack(side="left")
        ttk.Radiobutton(top, text="Screen", variable=self.capture_mode, value="screen").pack(side="left")
        ttk.Radiobutton(top, text="Camera", variable=self.capture_mode, value="camera").pack(side="left")
        # Initialize Input replaces the old Pick Region button.  Press this to
        # start the capture and OCR threads.  The old pick_region functionality
        # is still available via code but not exposed in the UI because the user
        # does not need region selection.
        ttk.Button(top, text="Initialize Input", command=self.init_input).pack(side="left", padx=6)
        ttk.Spinbox(top, from_=0, to=9, textvariable=self.camera_index, width=3).pack(side="left", padx=6)
        ttk.Button(top, text="Start", command=self.on_start).pack(side="left", padx=6)
        # Pause/Resume button â€“ store a reference so we can update its label
        # dynamically when paused or resumed.  The label is initially
        # "Pause (Space)" because capture starts only after initialisation.
        self.btn_pause = ttk.Button(top, text="Pause (Space)", command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=6)
        ttk.Button(top, text="Set Tesseractâ€¦", command=self.pick_tesseract).pack(side="left", padx=6)
        ttk.Button(top, text="Output Dirâ€¦", command=self.pick_outdir).pack(side="left", padx=6)

        # Profile save/load buttons. These allow the operator to persist
        # the entire capture/ROI configuration (including gamma, contrast,
        # thresholds, morphological parameters, ROI rectangles, screen box,
        # capture mode, camera index, tesseract path and output directory)
        # to a JSON file. A saved profile can later be reloaded to
        # immediately restore a previously tuned setup. The buttons are
        # placed on the top bar for easy access during setup.
        ttk.Button(top, text="Save Profileâ€¦", command=self.save_profile).pack(side="left", padx=6)
        ttk.Button(top, text="Load Profileâ€¦", command=self.load_profile).pack(side="left", padx=6)
        # Output format selector.  Allows the operator to choose whether to
        # write the clock values as plain text, JSON, or CSV.  Each value
        # will be written to its own file (game.<fmt>, play.<fmt>).  The
        # chosen format is persisted in the profile.
        ttk.Label(top, text="Format:").pack(side="left")
        fmt_combo = ttk.Combobox(top, textvariable=self.out_format, values=["txt","json","csv"], state="readonly", width=5)
        fmt_combo.pack(side="left", padx=6)

        self.lbl_now = ttk.Label(top, text="GAME: --:--    PLAY: --", font=("Segoe UI", 16, "bold"))
        self.lbl_now.pack(side="right")

        mid = ttk.Frame(root); mid.pack(fill="both", expand=True)
        # Viewer
        self.canvas = tk.Canvas(mid, bg="black", width=960, height=540, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(6,3), pady=6)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        root.bind("<space>", lambda e: self.toggle_pause())
        root.bind("a", lambda e: self.on_autocal())

        # Controls (scrollable)
        right = ttk.Frame(mid); right.pack(side="left", fill="y", padx=(3,6), pady=6)
        self.controls = self._build_controls(right)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status, anchor="w").pack(side="bottom", fill="x")

    def _build_controls(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill="y", expand=False)
        # Target selector
        ttk.Label(container, text="Active ROI Target:").pack(anchor="w", pady=(0,4))
        ttk.Radiobutton(container, text="GAME (M:SS)", variable=self.current_target, value="GAME").pack(anchor="w")
        ttk.Radiobutton(container, text="PLAY (SS)", variable=self.current_target, value="PLAY").pack(anchor="w")

        ttk.Separator(container).pack(fill="x", pady=6)

        # Legacy Red LED toggle for PLAY
        self.var_play_legacy = tk.BooleanVar(value=self.play_cfg.legacy_red_led)
        def on_legacy_change():
            self.play_cfg.legacy_red_led = self.var_play_legacy.get()
        ttk.Checkbutton(container, text="PLAY: Legacy Red LED", variable=self.var_play_legacy, command=on_legacy_change).pack(anchor="w")

        # Debloom toggle for PLAY.  When enabled, additional morphological
        # processing is applied to reduce bloom/halation artifacts that merge
        # adjacent LED segments.  This is particularly useful when the
        # play clock is distant in the frame and optical blur causes the
        # red lines to join together.  Changing this toggle updates
        # self.play_cfg.debloom immediately.
        self.var_play_debloom = tk.BooleanVar(value=getattr(self.play_cfg, 'debloom', False))
        def on_debloom_change():
            self.play_cfg.debloom = self.var_play_debloom.get()
        ttk.Checkbutton(container, text="PLAY: Debloom (far)", variable=self.var_play_debloom, command=on_debloom_change).pack(anchor="w")

        # Preprocess sliders grouped
        def add_slider(label, var_from, var_to, init, step, setter):
            frm = ttk.Frame(container); frm.pack(fill="x", pady=2)
            ttk.Label(frm, text=label, width=18).pack(side="left")
            val = tk.DoubleVar(value=init)
            s = ttk.Scale(frm, from_=var_from, to=var_to, orient="horizontal",
                          command=lambda e: setter(float(val.get())),
                          variable=val)
            s.pack(side="left", fill="x", expand=True, padx=6)
            ent = ttk.Entry(frm, width=6, textvariable=val)
            ent.pack(side="right")
            return val

        add_slider("Gamma", 0.5, 2.0, self.game_cfg.gamma, 0.05, lambda v: self._set_cfg("gamma", v))
        add_slider("Contrast", 0.5, 2.0, self.game_cfg.contrast, 0.05, lambda v: self._set_cfg("contrast", v))
        add_slider("Brightness", -50, 50, self.game_cfg.brightness, 1, lambda v: self._set_cfg("brightness", int(v)))

        # Threshold mode
        ttk.Label(container, text="Threshold").pack(anchor="w", pady=(6,2))
        self.var_thr_mode = tk.StringVar(value=self.game_cfg.thr_mode)
        for m in ("adaptive","otsu","fixed"):
            ttk.Radiobutton(container, text=m, variable=self.var_thr_mode, value=m,
                            command=lambda: self._set_cfg("thr_mode", self.var_thr_mode.get())).pack(anchor="w")
        self.var_thr_val = tk.IntVar(value=self.game_cfg.thr)
        frm_thr = ttk.Frame(container); frm_thr.pack(fill="x", pady=2)
        ttk.Label(frm_thr, text="Fixed thr").pack(side="left")
        ttk.Scale(frm_thr, from_=60, to=220, orient="horizontal",
                  variable=self.var_thr_val,
                  command=lambda e: self._set_cfg("thr", int(self.var_thr_val.get()))).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Entry(frm_thr, width=6, textvariable=self.var_thr_val).pack(side="right")

        # Morph
        ttk.Label(container, text="Morph (open/close)").pack(anchor="w", pady=(6,2))
        self.var_open = tk.IntVar(value=self.game_cfg.morph_open)
        self.var_close = tk.IntVar(value=self.game_cfg.morph_close)
        frm_m = ttk.Frame(container); frm_m.pack(fill="x", pady=2)
        ttk.Scale(frm_m, from_=1, to=7, variable=self.var_open, orient="horizontal",
                  command=lambda e: self._set_cfg("morph_open", int(self.var_open.get()))).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Scale(frm_m, from_=1, to=7, variable=self.var_close, orient="horizontal",
                  command=lambda e: self._set_cfg("morph_close", int(self.var_close.get()))).pack(side="left", fill="x", expand=True, padx=6)

        # Calibration controls â€“ embed expected value input and run button.  The user types
        # the expected clock value here and either clicks â€œRun Autoâ€‘Calâ€ or presses
        # the 'a' key to start calibration. This avoids modal dialogs and provides
        # better feedback during the calibration process.
        ttk.Separator(container).pack(fill="x", pady=6)
        ttk.Label(container, text="Autoâ€‘Cal (expected value)").pack(anchor="w")
        self.var_calib_expected = tk.StringVar(value="")
        self.entry_calib = ttk.Entry(container, textvariable=self.var_calib_expected)
        self.entry_calib.pack(fill="x")
        ttk.Button(container, text="Run Autoâ€‘Cal", command=self.on_autocal_from_entry).pack(fill="x", pady=3)
        # ROI clearing controls
        ttk.Button(container, text="Clear Active ROI (rightâ€‘click)", command=self.clear_active_roi).pack(fill="x", pady=(6,0))
        ttk.Button(container, text="Reset All ROIs", command=self.clear_all_rois).pack(fill="x")
        return container

    def _set_cfg(self, name, val):
        cfg = self.game_cfg if self.current_target.get()=="GAME" else self.play_cfg
        setattr(cfg, name, val)

    def pick_region(self):
        if not HAVE_MSS:
            messagebox.showerror("Screen capture", "mss not installed")
            return

        # Fullscreen translucent overlay
        picker = tk.Toplevel(self.root)
        picker.overrideredirect(True)
        picker.attributes("-topmost", True)
        try:
            picker.attributes("-alpha", 0.25)
        except Exception:
            pass

        sw = picker.winfo_screenwidth()
        sh = picker.winfo_screenheight()

        # Draw on a Canvas (Toplevel itself canâ€™t draw)
        canv = tk.Canvas(picker, bg="gray", highlightthickness=0, cursor="crosshair",
                        width=sw, height=sh)
        canv.pack(fill="both", expand=True)

        start = [0, 0]
        rect_id = [None]

        def md(e):
            start[0], start[1] = e.x, e.y
            if rect_id[0]:
                canv.delete(rect_id[0])
            rect_id[0] = canv.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=3)

        def mm(e):
            if rect_id[0] is not None:
                canv.coords(rect_id[0], start[0], start[1], e.x, e.y)

        def mu(e):
            x0, y0 = start
            x1, y1 = e.x, e.y
            left, top = min(x0, x1), min(y0, y1)
            w, h   = abs(x1 - x0), abs(y1 - y0)
            self.screen_box = {
                "left": int(left),
                "top": int(top),
                "width": int(max(1, w)),
                "height": int(max(1, h)),
            }
            picker.destroy()
            self.status.set(f"Screen region set: {self.screen_box}")

        canv.bind("<ButtonPress-1>", md)
        canv.bind("<B1-Motion>", mm)
        canv.bind("<ButtonRelease-1>", mu)


    def pick_tesseract(self):
        path = filedialog.askopenfilename(title="Locate tesseract executable")
        if path:
            self.tess_path.set(path)
            if HAVE_TESS:
                pytesseract.pytesseract.tesseract_cmd = path

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Choose output directory")
        if d:
            self.out_dir.set(d)

    def save_profile(self):
        """
        Save the current application settings to a JSON file.  This includes
        capture mode (screen/camera) and camera index, screen capture box,
        tesseract executable path, output directory, the perâ€‘ROI configuration
        parameters (gamma, contrast, brightness, threshold settings, morphological
        options, expected length, etc.), and the coordinates of each ROI.  The
        resulting JSON can be loaded later to restore the identical setup.
        """
        # Assemble all state into a plain serialisable structure
        profile = {
            "capture_mode": self.capture_mode.get(),
            "camera_index": int(self.camera_index.get()),
            "screen_box": self.screen_box,
            "tess_path": self.tess_path.get(),
            "out_dir": self.out_dir.get(),
            "game_cfg": vars(self.game_cfg),
            "play_cfg": vars(self.play_cfg),
            "game_rect": self.game_state.rect,
            "play_rect": self.play_state.rect,
            "out_format": self.out_format.get(),
        }
        # Prompt user to choose a file
        path = filedialog.asksaveasfilename(
            title="Save Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            self.status.set(f"Profile saved to {path}")
        except Exception as e:
            self.status.set(f"Failed to save profile: {e}")

    def load_profile(self):
        """
        Load application settings from a previously saved profile JSON.  This will
        restore capture mode, camera index, screen box, tesseract path, output
        directory, perâ€‘ROI configuration, and ROI coordinates.  After loading,
        the controls pane is rebuilt to reflect the new configuration values.
        """
        path = filedialog.askopenfilename(
            title="Load Profile",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.status.set(f"Failed to read profile: {e}")
            return
        try:
            # Restore simple values
            if "capture_mode" in data:
                self.capture_mode.set(data["capture_mode"])
            if "camera_index" in data:
                self.camera_index.set(int(data["camera_index"]))
            # Screen box may be None or a dict
            self.screen_box = data.get("screen_box")
            if "tess_path" in data and data["tess_path"]:
                self.tess_path.set(data["tess_path"])
                if HAVE_TESS:
                    pytesseract.pytesseract.tesseract_cmd = data["tess_path"]
            if "out_dir" in data and data["out_dir"]:
                self.out_dir.set(data["out_dir"])
            # Restore ROI configs
            if "game_cfg" in data:
                self.game_cfg = ROIConfig(**data["game_cfg"])
            if "play_cfg" in data:
                self.play_cfg = ROIConfig(**data["play_cfg"])
            # Restore ROI rectangles
            self.game_state.rect = tuple(data.get("game_rect")) if data.get("game_rect") else None
            self.play_state.rect = tuple(data.get("play_rect")) if data.get("play_rect") else None
            # Reset histories and last values because profile is loaded fresh
            self.game_state.history.clear(); self.game_state.last_value = ""
            self.play_state.history.clear(); self.play_state.last_value = ""
            # Restore output format
            if "out_format" in data and data["out_format"] in ("txt","json","csv"):
                try:
                    self.out_format.set(data["out_format"])
                except Exception:
                    pass
            # Rebuild controls to reflect new configuration
            try:
                # Destroy old controls frame
                if hasattr(self, "controls") and self.controls is not None:
                    parent = self.controls.master
                    self.controls.destroy()
                    # Build new controls in the same parent
                    self.controls = self._build_controls(parent)
            except Exception:
                # If rebuilding fails, ignore â€“ UI may not reflect loaded values
                pass
            self.status.set(f"Profile loaded from {path}")
        except Exception as e:
            self.status.set(f"Failed to apply profile: {e}")

    def init_input(self):
        """
        Initialize the capture and OCR threads.  This method is called when
        the user presses the "Initialize Input" button.  If threads are
        already running, it simply updates the paused flag and status; if
        threads have not yet been started, it spawns them.
        """
        # If threads already started, just unpause (if needed) and update
        # status.  Dynamic reinitialisation of capture will happen in
        # `_capture_loop` when the capture mode, camera index or region
        # changes.
        if getattr(self, 'threads_started', False):
            # Unpause and update status
            self.paused = False
            self.status.set("Capturingâ€¦")
            try:
                self.btn_pause.config(text="Pause (Space)")
            except Exception:
                pass
            return
        # First time initialisation: clear any previous stop flag, start
        # capture/ocr threads and mark threads as started
        try:
            # Ensure stop flag is reset
            self.stop_flag.clear()
        except Exception:
            # stop_flag may not exist if manually cleared; ignore
            pass
        self._start_threads()
        self.threads_started = True
        self.paused = False
        self.status.set("Capturingâ€¦")
        # Update pause button label to "Pause"
        try:
            self.btn_pause.config(text="Pause (Space)")
        except Exception:
            pass

    def toggle_pause(self):
        self.paused = not self.paused

    def on_start(self):
        """
        Start capture and OCR if not already started.  This method simply
        delegates to init_input(), which handles starting the threads and
        updating the status.  It remains available for compatibility.
        """
        self.init_input()

    def clear_active_roi(self):
        target = self.current_target.get()
        (self.game_state if target=="GAME" else self.play_state).rect = None

    # Mouse ROI
    def on_mouse_down(self, e):
        self._drag_start = (e.x, e.y)
        self._drag_rect = None

    def on_mouse_drag(self, e):
        if not hasattr(self, "_drag_start"): return
        x0,y0 = self._drag_start
        x1,y1 = e.x, e.y
        x,y = min(x0,x1), min(y0,y1)
        w,h = abs(x1-x0), abs(y1-y0)
        self._drag_rect = (x,y,w,h)

    def on_mouse_up(self, e):
        if getattr(self, "_drag_rect", None):
            # Map viewer coords to frame coords
            if hasattr(self, "last_frame"):
                vis_w = self.canvas.winfo_width()
                vis_h = self.canvas.winfo_height()
                fh, fw = self.last_frame.shape[:2]
                scale = min(vis_w/fw, vis_h/fh)
                pad_x = (vis_w - fw*scale)/2
                pad_y = (vis_h - fh*scale)/2
                x,y,w,h = self._drag_rect
                fx = int((x - pad_x)/scale); fy = int((y - pad_y)/scale)
                fw2 = int(w/scale); fh2 = int(h/scale)
                rect = (clamp(fx,0,fw-1), clamp(fy,0,fh-1), clamp(fw2,1,fw), clamp(fh2,1,fh))
                if self.current_target.get()=="GAME":
                    self.game_state.rect = rect
                else:
                    self.play_state.rect = rect
        self._drag_start = None
        self._drag_rect = None

    def on_right_click(self, e):
        self.clear_active_roi()

    def on_autocal(self):
        """Keybinding for Autoâ€‘Cal (pressing 'a'). Use entry field for expected value."""
        self.on_autocal_from_entry()

    def on_autocal_from_entry(self):
        """Perform autoâ€‘calibration using the value from the calibration entry with debug info.

        This function reads the expected clock value from the input field, checks that a
        frame and ROI are available, then performs a parameter sweep to calibrate
        preprocessing settings. During calibration, status messages are displayed in
        the status bar. After calibration, the function applies the new settings and
        verifies whether the OCR output matches the expected value. It reports
        success or failure along with suggestions on how to improve detection.
        """
        # Ensure a frame is available
        if getattr(self, "last_frame", None) is None:
            self.status.set("Auto-Cal: no frame available.")
            return
        # Determine current target and associated state/config
        target = self.current_target.get()
        state = self.game_state if target == "GAME" else self.play_state
        cfg = self.game_cfg if target == "GAME" else self.play_cfg
        # ROI must be defined
        if not state.rect:
            self.status.set("Auto-Cal: no ROI selected. Draw ROI first.")
            return
        # Get expected value from entry
        expected = self.var_calib_expected.get().strip()
        if not expected:
            self.status.set("Auto-Cal: enter expected value before calibrating.")
            return
        # Inform user that calibration is starting
        self.status.set(f"Auto-Cal: calibrating {target} (expected {expected})â€¦")
        # Force UI to update so message appears before blocking operations
        self.root.update_idletasks()
        # Extract ROI crop
        x, y, w, h = state.rect
        crop = self.last_frame[y:y + h, x:x + w].copy()
        # Run auto calibration
        t0 = time.time()
        newcfg = auto_calibrate(crop, cfg, target, expected)
        dt = (time.time() - t0) * 1000.0
        # Apply new configuration
        if target == "GAME":
            self.game_cfg = newcfg
        else:
            self.play_cfg = newcfg
        # Test the new configuration by re-reading the crop
        gray, binimg = apply_preprocess(crop, newcfg)
        predicted = read_game_clock(gray, binimg, newcfg) if target == "GAME" else read_play_clock(gray, binimg, newcfg)
        # Normalisation helpers
        def normalise_game(s: str) -> str:
            parts = s.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                m = str(int(parts[0]))  # remove leading zeros
                sec = parts[1].rjust(2, "0")[:2]
                return f"{m}:{sec}"
            return s
        def normalise_play(s: str) -> str:
            return s.zfill(2) if s.isdigit() else s
        # Determine success
        if target == "GAME":
            success = normalise_game(predicted) == normalise_game(expected)
        else:
            success = normalise_play(predicted) == normalise_play(expected)
        # Provide feedback
        if success:
            self.status.set(f"Auto-Cal: success in {dt:.0f} ms. Result: {predicted}.")
        else:
            self.status.set(f"Auto-Cal: finished in {dt:.0f} ms but got '{predicted}' (expected {expected}). "
                            "Try redrawing the ROI tightly around the digits or adjusting brightness/contrast.")
        # Update UI elements
        self.root.update_idletasks()

    def clear_all_rois(self):
        """Clear both GAME and PLAY ROIs."""
        self.game_state.rect = None
        self.play_state.rect = None

    # ---------- Threads ----------
    def _start_threads(self):
        self.cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
        self.cap_thread.start()
        self.ocr_thread.start()

    def _make_capture(self) -> Optional[CaptureSource]:
        if self.args.test_image:
            return None
        mode = self.capture_mode.get()
        cs = CaptureSource(mode=mode,
                           camera_index=int(self.camera_index.get()),
                           screen_box=self.screen_box,
                           desired_size=(1920,1080))
        try:
            ok = cs.open()
        except Exception as e:
            try:
                self.status.set(f"Failed to open capture: {e}")
            except Exception:
                pass
            return None
        if not ok:
            try:
                self.status.set(f"Failed to open camera index {self.camera_index.get()}")
            except Exception:
                pass
            return None
        return cs

    def _capture_loop(self):
        """Background thread: capture frames from the chosen input.

        This loop dynamically reinitialises the capture source whenever the
        capture mode (screen/camera), camera index, or screen box changes.
        It also avoids holding a stale handle to a previous input by
        releasing the old capture before opening a new one. When in test
        image mode, it simply reuses the static image.
        """
        cs = None
        img_static = None
        # If a test image is provided, load it once and skip capture
        if self.args.test_image:
            if not os.path.isfile(self.args.test_image):
                self.status.set(f"Test image not found: {self.args.test_image}")
                return
            bgr = cv2.imread(self.args.test_image, cv2.IMREAD_COLOR)
            if bgr is None:
                self.status.set("Failed to read test image.")
                return
            img_static = bgr
        # Reset last capture configuration so the first iteration opens the source
        self._capture_last_mode = None
        self._capture_last_index = None
        self._capture_last_box = None
        while not self.stop_flag.is_set():
            if self.paused:
                time.sleep(0.03)
                continue
            # Reload static image each iteration if test_image mode
            if img_static is not None:
                frame = img_static.copy()
            else:
                # Determine current desired capture configuration
                current_mode = self.capture_mode.get()
                current_index = int(self.camera_index.get())
                current_box = self.screen_box
                # If no capture or configuration has changed, reinitialise
                if (cs is None or
                    current_mode != self._capture_last_mode or
                    current_index != self._capture_last_index or
                    current_box != self._capture_last_box):
                    # Release previous capture if any
                    if cs is not None:
                        try:
                            cs.release()
                        except Exception:
                            pass
                    # Create new capture source
                    try:
                        cs = self._make_capture()
                    except Exception as e:
                        # Capture creation failed; show error and retry later
                        self.status.set(f"Failed to open {current_mode} source: {e}")
                        cs = None
                    # Update last-known configuration so we won't re-open again until changed
                    self._capture_last_mode = current_mode
                    self._capture_last_index = current_index
                    self._capture_last_box = current_box
                # Read a frame if capture is available
                if cs is None:
                    # No capture available; wait and retry
                    time.sleep(0.03)
                    continue
                try:
                    frame = cs.read()
                except Exception:
                    frame = None
            # If no frame, wait briefly
            if frame is None:
                time.sleep(0.01)
                continue
            # Store last frame for ROI mapping and auto-calibration
            self.last_frame = frame
            # Push to queue (drop oldest if queue is full)
            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass
            # Schedule viewer update on the main thread
            fcopy = frame.copy()
            try:
                self.root.after(0, lambda f=fcopy: self._update_viewer(f))
            except Exception:
                self._update_viewer(fcopy)
        # On exit, release capture if any
        if cs:
            try:
                cs.release()
            except Exception:
                pass

    def _update_viewer(self, frame):
        vis_w = self.canvas.winfo_width()
        vis_h = self.canvas.winfo_height()
        fh, fw = frame.shape[:2]
        if fw == 0 or fh == 0:
            return
        scale = min(vis_w/fw, vis_h/fh)
        # resized display image
        disp_w, disp_h = int(fw*scale), int(fh*scale)
        disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # draw scaled ROIs onto disp (so they are visible and in-sync)
        if self.game_state.rect:
            x,y,w,h = self.game_state.rect
            sx, sy = int(x*scale), int(y*scale)
            sw, sh = max(1,int(w*scale)), max(1,int(h*scale))
            cv2.rectangle(disp, (sx, sy), (sx+sw, sy+sh), (50,220,50), 2, cv2.LINE_AA)
        if self.play_state.rect:
            x,y,w,h = self.play_state.rect
            sx, sy = int(x*scale), int(y*scale)
            sw, sh = max(1,int(w*scale)), max(1,int(h*scale))
            cv2.rectangle(disp, (sx, sy), (sx+sw, sy+sh), (220,220,50), 2, cv2.LINE_AA)

        # compute padding used to center the disp inside the canvas
        pad_x = int((vis_w - disp_w) / 2)
        pad_y = int((vis_h - disp_h) / 2)

        # also show drag rect on the displayed image (convert viewer coords -> disp coords)
        if getattr(self, "_drag_rect", None):
            x,y,w,h = self._drag_rect  # these are viewer coords
            rx = int(x - pad_x); ry = int(y - pad_y)
            rx2 = rx + int(w); ry2 = ry + int(h)
            # clamp to disp bounds
            rx = clamp(rx, 0, disp_w); ry = clamp(ry, 0, disp_h)
            rx2 = clamp(rx2, 0, disp_w); ry2 = clamp(ry2, 0, disp_h)
            if rx2 > rx and ry2 > ry:
                cv2.rectangle(disp, (rx, ry), (rx2, ry2), (0,255,255), 2, cv2.LINE_AA)

        # convert & update the single canvas image item (keep PhotoImage ref to avoid GC)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._tk_im = ImageTk.PhotoImage(Image.fromarray(rgb))

        # image center in canvas = pad + half of disp (ensures correct placement)
        cx = pad_x + disp_w//2
        cy = pad_y + disp_h//2
        if self._canvas_img_id is None:
            # create once, centered at computed coords
            self._canvas_img_id = self.canvas.create_image(cx, cy, image=self._tk_im)
        else:
            # update existing image item and its position (prevents stacking and fixes offset)
            try:
                self.canvas.itemconfig(self._canvas_img_id, image=self._tk_im)
                self.canvas.coords(self._canvas_img_id, cx, cy)
            except Exception:
                # in rare cases item may be gone; recreate
                self._canvas_img_id = self.canvas.create_image(cx, cy, image=self._tk_im)

    def _ocr_loop(self):
        # Output write timing and current values.  Each value will be
        # written to its own file (game.<fmt> and play.<fmt>) based on
        # self.out_format.  Initialize last write time to zero and default
        # display values.
        t_last_write = 0
        game_val, play_val = "--:--", "--"
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            # GAME
            if self.game_state.rect:
                x,y,w,h = self.game_state.rect
                crop = frame[y:y+h, x:x+w]
                gray, binimg = apply_preprocess(crop, self.game_cfg)
                val = read_game_clock(gray, binimg, self.game_cfg)
                if val:
                    # Determine if we should run block detection.  We only run
                    # this costly operation when the new value differs from
                    # the last accepted value by more than 5 seconds.  This
                    # avoids unnecessary ratio calculations when the clock
                    # changes normally (counting down by one second).
                    accept = True
                    # Helper to convert "M:SS" strings to seconds
                    def _parse_time(s: str) -> Optional[int]:
                        try:
                            if ":" in s:
                                m, ss = s.split(":", 1)
                                if m.isdigit() and ss.isdigit():
                                    return int(m) * 60 + int(ss)
                        except Exception:
                            pass
                        return None
                    last_val = self.game_state.last_value
                    if last_val:
                        last_sec = _parse_time(last_val)
                        new_sec = _parse_time(val)
                        if last_sec is not None and new_sec is not None:
                            if abs(new_sec - last_sec) > 5:
                                # Compute white pixel ratio to decide if the
                                # clock is covered.  We invert binimg in
                                # apply_preprocess only when needed; here we
                                # simply compute fraction of 'on' pixels.
                                ratio = float(np.mean(binimg > 0))
                                # If ratio is extremely low or extremely high,
                                # the digits are likely covered by a hand or
                                # graphic.  In that case, skip updating the
                                # clock and retain the last value.
                                if ratio < 0.02 or ratio > 0.60:
                                    accept = False
                    if accept:
                        # append and update last_value
                        self.game_state.history.append(val)
                        self.game_state.last_value = val
                        # Smooth the history to reduce jitter
                        game_val = median_smooth(self.game_state.history, k=3)
                    else:
                        # retain last value by not appending new val
                        game_val = self.game_state.last_value or game_val
                # If val is empty we do nothing (retain previous game_val)
            # PLAY
            if self.play_state.rect:
                x,y,w,h = self.play_state.rect
                crop = frame[y:y+h, x:x+w]
                gray, binimg = apply_preprocess(crop, self.play_cfg)
                val = read_play_clock(gray, binimg, self.play_cfg)
                if val:
                    # update history and last value for play clock
                    self.play_state.history.append(val)
                    self.play_state.last_value = val
                    play_val = median_smooth(self.play_state.history, k=3)
            # Guards: game should count down or hold; play 40->39 or 25->24 etc (lightweight â€“ not enforced hard)

            self.lbl_now.config(text=f"GAME: {game_val:>5}    PLAY: {play_val:>2}")
            # Write outputs at ~10 Hz
            now = time.time()
            if now - t_last_write > 0.1:
                # Determine output format and write each value separately
                fmt = self.out_format.get().lower()
                out_dir = self.out_dir.get()
                if fmt == "txt":
                    atomic_write_text(os.path.join(out_dir, "game.txt"), f"{game_val}\n")
                    atomic_write_text(os.path.join(out_dir, "play.txt"), f"{play_val}\n")
                elif fmt == "json":
                    atomic_write_json(os.path.join(out_dir, "game.json"), {"value": game_val, "ts": now})
                    atomic_write_json(os.path.join(out_dir, "play.json"), {"value": play_val, "ts": now})
                elif fmt == "csv":
                    atomic_write_csv(os.path.join(out_dir, "game.csv"), [["value","ts"], [game_val, f"{now:.3f}"]])
                    atomic_write_csv(os.path.join(out_dir, "play.csv"), [["value","ts"], [play_val, f"{now:.3f}"]])
                # unknown format fallback: write a combined text file
                else:
                    atomic_write_text(os.path.join(out_dir, "clock.txt"), f"{game_val} {play_val}\n")
                t_last_write = now

    # ---------- Cleanup ----------
    def close(self):
        self.stop_flag.set()
        self.root.after(150, self.root.destroy)

# ----------------------------- Main ---------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image", default=None, help="Path to a test still image to drive the viewer/ocr")
    args = parser.parse_args()

    if HAVE_TESS:
        # nothing â€“ user can override through UI
        pass

    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root, args)
    def on_close():
        app.close()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()