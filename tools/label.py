"""Browser-based frame-by-frame bounding-box labeler for tiny drones.

Zero extra dependencies: a stdlib HTTP server (``http.server``) serves a
single-page canvas app; frames are decoded once, sequentially, JPEG-encoded, and held in memory
for instant random access.

    .venv/bin/python tools/label.py                      # 07_05.mp4 -> work/labels.json
    .venv/bin/python tools/label.py --from-gt work/gt.json   # seed boxes from existing GT
    .venv/bin/python tools/label.py --export-gt work/gt.json # write labels back as GT and exit

Labels live in ``work/labels.json`` as per-frame corner boxes::

    {"video": "07_05.mp4", "width": 1280, "height": 720,
     "classes": ["far", "near", "bird"],
     "frames": {"0": [{"x1":.., "y1":.., "x2":.., "y2":.., "label":"far"}], ...}}

The UI autosaves; boxes are in original-image pixel coordinates, matching
``dronedet.gt`` / ``dronedet.detections``. Export groups boxes by label into
GT objects (``objects[label].frames[t] = [cx, cy, w, h]``), carrying over
``meta`` (e.g. stabilization shifts) from an existing GT file when present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.video import frames, probe

DEFAULT_CLASSES = ["far", "near", "bird"]

# Repo root, so relative paths (the video, work/*.json) resolve the same way
# whether the tool is launched from the root or from an IDE with a different cwd.
ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str | Path | None) -> Path | None:
    """Absolute paths are honored; a relative path is used as-is if it already
    exists in the cwd, otherwise anchored to the repo root."""
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute() or pp.exists():
        return pp
    return ROOT / pp

# ---------------------------------------------------------------------------
# Shared server state (guarded by STATE["lock"])
# ---------------------------------------------------------------------------
STATE: dict = {
    "jpegs": [],          # list[bytes] one JPEG per readable frame
    "info": {},           # {video, n, width, height}
    "labels": {},         # {"video","width","height","classes","frames":{str:[box]}}
    "labels_path": None,  # Path
    "lock": threading.Lock(),
}


# ---------------------------------------------------------------------------
# Frame loading / label persistence
# ---------------------------------------------------------------------------
def load_frames(video: str, quality: int = 95) -> list[bytes]:
    """Decode the whole video sequentially into per-frame JPEG bytes."""
    info = probe(video)
    print(f"loading {video} ({info.width}x{info.height})...", flush=True)
    jpegs: list[bytes] = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for idx, frame in frames(video):
        ok, buf = cv2.imencode(".jpg", frame, enc)
        if not ok:
            raise RuntimeError(f"failed to JPEG-encode frame {idx}")
        jpegs.append(buf.tobytes())
        if idx % 100 == 0:
            print(f"  decoded {idx} frames", flush=True)
    print(f"loaded {len(jpegs)} frames", flush=True)
    return jpegs


def gt_to_labels(gt_path: str | Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Convert a GT file's objects into per-frame corner boxes.

    Returns (frames_dict, class_names). GT stores (cx, cy, w, h); we emit
    (x1, y1, x2, y2) tagged with the object name as the class label.
    """
    raw = json.loads(Path(gt_path).read_text())
    per_frame: dict[str, list[dict]] = {}
    classes: list[str] = []
    for name, obj in raw.get("objects", {}).items():
        classes.append(name)
        for f, box in obj.get("frames", {}).items():
            cx, cy, w, h = box[:4]
            per_frame.setdefault(str(int(f)), []).append({
                "x1": round(cx - w / 2, 2), "y1": round(cy - h / 2, 2),
                "x2": round(cx + w / 2, 2), "y2": round(cy + h / 2, 2),
                "label": name,
            })
    return per_frame, classes


def labels_to_gt(labels: dict, gt_path: str | Path) -> None:
    """Write per-frame boxes back into GT object format.

    Boxes are grouped by label; a label carrying multiple boxes in the same
    frame is split into ``label#2``, ``label#3`` slots (stable by position).
    ``meta`` from an existing GT at the same path is preserved (shifts etc.).
    """
    gt_path = Path(gt_path)
    meta: dict = {}
    prev_ignore: dict[str, bool] = {}
    if gt_path.exists():
        old = json.loads(gt_path.read_text())
        meta = old.get("meta", {})
        prev_ignore = {n: o.get("ignore", False)
                       for n, o in old.get("objects", {}).items()}

    objects: dict[str, dict[str, list]] = {}
    for f, boxes in labels.get("frames", {}).items():
        counts: dict[str, int] = {}
        for b in boxes:
            base = b.get("label", "obj")
            counts[base] = counts.get(base, 0) + 1
            name = base if counts[base] == 1 else f"{base}#{counts[base]}"
            cx = (b["x1"] + b["x2"]) / 2.0
            cy = (b["y1"] + b["y2"]) / 2.0
            w = abs(b["x2"] - b["x1"])
            h = abs(b["y2"] - b["y1"])
            objects.setdefault(name, {})[str(int(f))] = [
                round(cx, 2), round(cy, 2), round(w, 2), round(h, 2)]

    payload = {
        "video": labels.get("video", ""),
        "meta": meta,
        "objects": {
            name: {"ignore": prev_ignore.get(name.split("#")[0], False),
                   "frames": dict(sorted(fr.items(), key=lambda kv: int(kv[0])))}
            for name, fr in objects.items()
        },
    }
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(payload))
    print(f"exported {len(objects)} objects to {gt_path}")


def save_labels() -> None:
    """Atomically persist STATE['labels'] to disk (caller holds the lock)."""
    path: Path = STATE["labels_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(STATE["labels"]))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/info":
            with STATE["lock"]:
                self._json(200, {
                    "info": STATE["info"],
                    "classes": STATE["labels"].get("classes", DEFAULT_CLASSES),
                    "labels": STATE["labels"].get("frames", {}),
                })
            return
        if path.startswith("/api/frame/"):
            try:
                i = int(path.rsplit("/", 1)[1].split(".")[0])
                if i < 0:
                    raise IndexError  # -1 must 404, not wrap to the last frame
                body = STATE["jpegs"][i]
            except (ValueError, IndexError):
                self._send(404, b"no such frame", "text/plain")
                return
            self._send(200, body, "image/jpeg", cache=True)
            return
        self._send(404, b"not found", "text/plain")

    def _read_body(self) -> bytes:
        """Read exactly Content-Length bytes (always, so the next request on a
        keep-alive connection stays framed); tolerate a bad/absent length."""
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            n = 0
        return self.rfile.read(n) if n > 0 else b""

    def _read_json(self, default):
        """Drain + parse the body; returns (obj, ok). The body is always
        consumed before we reply, even on a parse error, to avoid keep-alive
        request desync."""
        raw = self._read_body()
        if not raw:
            return default, True
        try:
            return json.loads(raw), True
        except (ValueError, TypeError):
            return default, False

    def _save_frame(self, i: int, boxes) -> None:
        with STATE["lock"]:
            fr = STATE["labels"].setdefault("frames", {})
            if boxes:
                fr[str(i)] = boxes
            else:
                fr.pop(str(i), None)
            save_labels()

    def _labels_index(self, path: str):
        try:
            return int(path.rsplit("/", 1)[1]), True
        except ValueError:
            return -1, False

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/labels/"):
            boxes, ok = self._read_json([])          # drain body first
            i, ok_i = self._labels_index(path)
            if not ok_i:
                self._send(400, b"bad frame index", "text/plain")
                return
            if not ok:
                self._send(400, b"bad json", "text/plain")
                return
            self._save_frame(i, boxes)
            self._json(200, {"ok": True})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        # sendBeacon (fired on tab close) can only POST -> accept the save here
        if path.startswith("/api/labels/"):
            boxes, ok = self._read_json([])
            i, ok_i = self._labels_index(path)
            if ok and ok_i:
                self._save_frame(i, boxes)
            self._json(200, {"ok": ok and ok_i})
            return
        if path == "/api/classes":
            classes, ok = self._read_json([])
            if ok:
                with STATE["lock"]:
                    STATE["labels"]["classes"] = classes
                    save_labels()
            self._json(200 if ok else 400, {"ok": ok})
            return
        if path == "/api/export_gt":
            body, _ = self._read_json({})
            out = _resolve((body or {}).get("path") or "work/gt_labeled.json")
            with STATE["lock"]:
                labels_to_gt(STATE["labels"], out)
            self._json(200, {"ok": True, "path": str(out)})
            return
        self._send(404, b"not found", "text/plain")


# ---------------------------------------------------------------------------
# Single-page app
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>drone labeler</title>
<style>
  :root { --bg:#14161a; --panel:#1d2027; --line:#2c313c; --fg:#e6e9ef;
          --mut:#8a92a6; --accent:#4c9aff; --ok:#3ecf8e; --warn:#ffb020; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
              font:13px/1.4 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;
              overflow:hidden; -webkit-user-select:none; user-select:none; }
  #app { display:grid; grid-template-rows:auto 1fr auto; height:100%; }
  header { display:flex; align-items:center; gap:10px; padding:6px 10px;
           background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header .grp { display:flex; align-items:center; gap:6px; }
  header .sep { width:1px; height:22px; background:var(--line); margin:0 2px; }
  button { background:#272b34; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 9px; cursor:pointer; font-size:12px; }
  button:hover { background:#30353f; }
  button.on { background:var(--accent); border-color:var(--accent); color:#04101f; font-weight:600; }
  input[type=number], input[type=text] { background:#0e1014; color:var(--fg);
           border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
  .title { font-weight:700; letter-spacing:.3px; }
  .mut { color:var(--mut); }
  #stage { position:relative; overflow:hidden; background:#0a0b0d; }
  #view { position:absolute; inset:0; width:100%; height:100%; display:block; cursor:crosshair; outline:none; }
  #loupe { position:absolute; right:12px; top:12px; width:200px; height:200px;
           border:1px solid var(--line); border-radius:8px; background:#000;
           box-shadow:0 6px 24px rgba(0,0,0,.5); pointer-events:none; opacity:0; transition:opacity .12s; }
  #loupe.show { opacity:1; }
  #classbar { position:absolute; left:12px; top:12px; display:flex; flex-direction:column;
              gap:6px; background:rgba(20,22,26,.82); padding:8px; border:1px solid var(--line);
              border-radius:8px; max-width:220px; }
  #classbar .row { display:flex; align-items:center; gap:6px; }
  .swatch { width:12px; height:12px; border-radius:3px; flex:none; }
  .cls { display:flex; align-items:center; gap:6px; padding:3px 6px; border-radius:6px;
         cursor:pointer; border:1px solid transparent; }
  .cls.active { border-color:var(--accent); background:#232a36; }
  .cls .k { color:var(--mut); font-variant-numeric:tabular-nums; width:12px; }
  #hint { position:absolute; left:12px; bottom:12px; pointer-events:none; background:rgba(20,22,26,.82);
          border:1px solid var(--line); border-radius:8px; padding:8px 10px; max-width:340px;
          color:var(--mut); font-size:11px; line-height:1.7; }
  #hint b { color:var(--fg); }
  footer { display:flex; align-items:center; gap:12px; padding:6px 10px; background:var(--panel);
           border-top:1px solid var(--line); }
  #scrub { flex:1; }
  #timeline { position:relative; height:14px; flex:2; background:#0e1014;
              border:1px solid var(--line); border-radius:4px; overflow:hidden; cursor:pointer; }
  .tick { position:absolute; top:0; bottom:0; width:2px; background:var(--ok); opacity:.55; }
  #cursorline { position:absolute; top:-2px; bottom:-2px; width:2px; background:var(--accent); }
  .pill { padding:2px 8px; border-radius:999px; background:#0e1014; border:1px solid var(--line);
          font-variant-numeric:tabular-nums; }
  #saved.dirty { color:var(--warn); }
  #saved.ok { color:var(--ok); }
  kbd { background:#0e1014; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:4px; padding:0 4px; font-size:10px; color:var(--fg); }
</style>
</head>
<body>
<div id="app">
  <header>
    <span class="title">🛰 drone labeler</span>
    <span class="sep"></span>
    <div class="grp">
      <button id="prev" title="prev frame (,)">◀</button>
      <input id="frameno" type="number" min="0" value="0" style="width:72px" title="go to frame">
      <span class="mut" id="framecnt">/ 0</span>
      <button id="next" title="next frame (.)">▶</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="copyprev" title="copy boxes from previous frame (c)">⧉ copy prev</button>
      <button id="carry" title="carry boxes onto next empty frame">carry ▶</button>
      <button id="clear" title="delete all boxes on this frame">✕ clear</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="zoomout" title="zoom out (-)">−</button>
      <span class="pill" id="zoomlbl">100%</span>
      <button id="zoomin" title="zoom in (+)">+</button>
      <button id="fit" title="fit (f)">fit</button>
      <button id="one" title="1:1 (0)">1:1</button>
    </div>
    <span class="sep"></span>
    <div class="grp">
      <button id="export" title="export to GT json">⬇ export GT</button>
      <span id="saved" class="pill ok">saved</span>
    </div>
  </header>

  <div id="stage">
    <canvas id="view" tabindex="0"></canvas>
    <canvas id="loupe" width="200" height="200"></canvas>
    <div id="classbar"></div>
    <div id="hint">
      <b>draw</b> drag on empty · <b>move</b> drag a box · <b>resize</b> drag a handle ·
      <b>pan</b> space-drag / middle-drag<br>
      <kbd>,</kbd><kbd>.</kbd> frame · <kbd>c</kbd> copy prev · <kbd>Del</kbd> delete ·
      <kbd>1</kbd>-<kbd>9</kbd> class · arrows nudge selected · <kbd>f</kbd> fit · <kbd>Esc</kbd> deselect
    </div>
  </div>

  <footer>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <div id="timeline"><div id="cursorline" style="left:0"></div></div>
    <span class="pill" id="posinfo">–</span>
    <span class="pill" id="boxinfo">0 boxes</span>
  </footer>
</div>

<script>
"use strict";
const $ = s => document.querySelector(s);
const view = $("#view"), vctx = view.getContext("2d");
const loupe = $("#loupe"), lctx = loupe.getContext("2d");

const PALETTE = ["#4c9aff","#3ecf8e","#ffb020","#ff6b6b","#c084fc",
                 "#22d3ee","#f472b6","#a3e635","#fb923c","#94a3b8"];

let INFO = {n:0, width:1280, height:720, video:""};
let CLASSES = [];
let LABELS = {};            // {frameIndexStr: [ {x1,y1,x2,y2,label} ]}
let cur = 0;               // current frame index
let boxes = [];            // boxes of the current frame (working copy)
let activeClass = 0;
let selected = -1;         // index into boxes
let carry = false;

// view transform: screen = image*scale + (ox,oy)
let scale = 1, ox = 0, oy = 0;
let img = new Image();
const imgCache = new Map();  // idx -> Image

// ---- helpers ----
function classColor(name){ const i = CLASSES.indexOf(name); return PALETTE[(i<0?0:i)%PALETTE.length]; }
function s2i(sx, sy){ return [(sx-ox)/scale, (sy-oy)/scale]; }
function i2s(ix, iy){ return [ix*scale+ox, iy*scale+oy]; }
function clampFrame(i){ return Math.max(0, Math.min(INFO.n-1, i|0)); }
function norm(b){ return {x1:Math.min(b.x1,b.x2), y1:Math.min(b.y1,b.y2),
                          x2:Math.max(b.x1,b.x2), y2:Math.max(b.y1,b.y2), label:b.label}; }

function resize(){
  const r = $("#stage").getBoundingClientRect();
  view.width = Math.round(r.width); view.height = Math.round(r.height);
  draw();
}

function fitView(){
  scale = Math.min(view.width/INFO.width, view.height/INFO.height);
  ox = (view.width - INFO.width*scale)/2;
  oy = (view.height - INFO.height*scale)/2;
  draw();
}

function zoomAt(sx, sy, factor){
  const [ix, iy] = s2i(sx, sy);
  scale = Math.max(0.05, Math.min(80, scale*factor));
  ox = sx - ix*scale; oy = sy - iy*scale;
  draw();
}

// ---- frame loading & prefetch ----
function frameImg(i){
  if(imgCache.has(i)) return imgCache.get(i);
  const im = new Image();
  im.src = `/api/frame/${i}.jpg`;
  imgCache.set(i, im);
  if(imgCache.size > 60){ const k = imgCache.keys().next().value; imgCache.delete(k); }
  return im;
}
function prefetch(i){ for(let d=-2; d<=4; d++){ const j=i+d; if(j>=0&&j<INFO.n) frameImg(j); } }

function loadFrame(i, {commit=true}={}){
  if(commit) flush();
  cur = clampFrame(i);
  const wasEmpty = !(LABELS[cur] && LABELS[cur].length);
  boxes = (LABELS[cur] || []).map(b => ({...b}));
  if(carry && wasEmpty && cur>0 && LABELS[cur-1]){
    boxes = LABELS[cur-1].map(b => ({...b}));
    markDirty();
  }
  selected = -1;
  img = frameImg(cur);
  if(img.complete) draw(); else img.onload = draw;
  prefetch(cur);
  syncUI();
}

// ---- drawing ----
function draw(){
  vctx.setTransform(1,0,0,1,0,0);
  vctx.clearRect(0,0,view.width,view.height);
  vctx.fillStyle = "#0a0b0d"; vctx.fillRect(0,0,view.width,view.height);
  vctx.imageSmoothingEnabled = false;
  if(img && img.complete && img.naturalWidth){
    vctx.drawImage(img, ox, oy, INFO.width*scale, INFO.height*scale);
  }
  // image border
  vctx.strokeStyle = "#2c313c"; vctx.lineWidth = 1;
  vctx.strokeRect(ox+.5, oy+.5, INFO.width*scale, INFO.height*scale);

  for(let k=0;k<boxes.length;k++){
    const b = norm(boxes[k]);
    const [x1,y1] = i2s(b.x1,b.y1), [x2,y2] = i2s(b.x2,b.y2);
    const col = classColor(b.label);
    vctx.lineWidth = k===selected ? 2 : 1.4;
    vctx.strokeStyle = col;
    vctx.strokeRect(x1, y1, x2-x1, y2-y1);
    // crosshair through center for pinpoint tiny boxes
    if(k===selected){
      const cx=(x1+x2)/2, cy=(y1+y2)/2;
      vctx.strokeStyle = col+"80"; vctx.lineWidth=1;
      vctx.beginPath(); vctx.moveTo(cx-8,cy); vctx.lineTo(cx+8,cy);
      vctx.moveTo(cx,cy-8); vctx.lineTo(cx,cy+8); vctx.stroke();
      for(const [hx,hy] of handlePts(b)){
        vctx.fillStyle = "#0a0b0d"; vctx.fillRect(hx-4,hy-4,8,8);
        vctx.strokeStyle = col; vctx.lineWidth=1.5; vctx.strokeRect(hx-4,hy-4,8,8);
      }
    }
    // label tag
    const tag = b.label;
    vctx.font = "11px ui-sans-serif";
    const tw = vctx.measureText(tag).width + 8;
    const ty = y1 > 14 ? y1-14 : y2+2;
    vctx.fillStyle = col; vctx.fillRect(x1, ty, tw, 13);
    vctx.fillStyle = "#04101f"; vctx.fillText(tag, x1+4, ty+10);
  }
}

function handlePts(b){  // 8 handles in screen space
  const [x1,y1]=i2s(b.x1,b.y1), [x2,y2]=i2s(b.x2,b.y2);
  const mx=(x1+x2)/2, my=(y1+y2)/2;
  return [[x1,y1,"nw"],[mx,y1,"n"],[x2,y1,"ne"],[x2,my,"e"],
          [x2,y2,"se"],[mx,y2,"s"],[x1,y2,"sw"],[x1,my,"w"]];
}

// ---- hit testing ----
function hitHandle(sx, sy){
  if(selected<0) return null;
  for(const [hx,hy,tag] of handlePts(norm(boxes[selected])))
    if(Math.abs(sx-hx)<=6 && Math.abs(sy-hy)<=6) return tag;
  return null;
}
function hitBox(sx, sy){
  const [ix,iy]=s2i(sx,sy); const tol=6/scale;
  for(let k=boxes.length-1;k>=0;k--){
    const b=norm(boxes[k]);
    if(ix>=b.x1-tol && ix<=b.x2+tol && iy>=b.y1-tol && iy<=b.y2+tol) return k;
  }
  return -1;
}

// ---- loupe (magnifier under cursor) ----
function drawLoupe(sx, sy){
  if(!img || !img.complete || !img.naturalWidth){ loupe.classList.remove("show"); return; }
  const [ix,iy]=s2i(sx,sy);
  if(ix<0||iy<0||ix>INFO.width||iy>INFO.height){ loupe.classList.remove("show"); return; }
  loupe.classList.add("show");
  // dock in the corner farthest from the cursor so it never occludes the
  // target (which enters from the top-right) or sits under the pointer.
  loupe.style.left = sx < view.width/2 ? "auto" : "12px";
  loupe.style.right = sx < view.width/2 ? "12px" : "auto";
  loupe.style.top = sy < view.height/2 ? "auto" : "12px";
  loupe.style.bottom = sy < view.height/2 ? "12px" : "auto";
  const mag=8, half=loupe.width/(2*mag);
  lctx.imageSmoothingEnabled=false;
  lctx.clearRect(0,0,loupe.width,loupe.height);
  lctx.drawImage(img, ix-half, iy-half, half*2, half*2, 0,0,loupe.width,loupe.height);
  // draw boxes in loupe space
  const L = (x,y)=>[(x-(ix-half))*mag, (y-(iy-half))*mag];
  for(const raw of boxes){ const b=norm(raw); const col=classColor(b.label);
    const [lx1,ly1]=L(b.x1,b.y1),[lx2,ly2]=L(b.x2,b.y2);
    lctx.strokeStyle=col; lctx.lineWidth=1.5; lctx.strokeRect(lx1,ly1,lx2-lx1,ly2-ly1); }
  // crosshair
  lctx.strokeStyle="#ffffff90"; lctx.lineWidth=1;
  lctx.beginPath(); lctx.moveTo(loupe.width/2,0); lctx.lineTo(loupe.width/2,loupe.height);
  lctx.moveTo(0,loupe.height/2); lctx.lineTo(loupe.width,loupe.height/2); lctx.stroke();
}

// ---- pointer interaction ----
let drag = null;  // {mode, sx0, sy0, ...}
view.addEventListener("mousedown", e=>{
  view.focus();
  const sx=e.offsetX, sy=e.offsetY;
  const panning = e.button===1 || spaceDown;
  if(panning){ drag={mode:"pan", sx0:sx, sy0:sy, ox0:ox, oy0:oy}; e.preventDefault(); return; }
  if(e.button!==0) return;
  const h = hitHandle(sx, sy);
  if(h){ drag={mode:"resize", handle:h, k:selected, box0:{...norm(boxes[selected])}}; return; }
  const k = hitBox(sx, sy);
  if(k>=0){
    selected=k;
    const [ix,iy]=s2i(sx,sy);
    drag={mode:"move", k, ix0:ix, iy0:iy, box0:{...norm(boxes[k])}};
    draw(); syncUI(); return;
  }
  // draw new
  const [ix,iy]=s2i(sx,sy);
  boxes.push({x1:ix,y1:iy,x2:ix,y2:iy,label:CLASSES[activeClass]});
  selected=boxes.length-1;
  drag={mode:"draw", k:selected};
  draw(); syncUI();
});

window.addEventListener("mousemove", e=>{
  const rect=view.getBoundingClientRect();
  const sx=e.clientX-rect.left, sy=e.clientY-rect.top;
  if(sx>=0&&sy>=0&&sx<=rect.width&&sy<=rect.height){
    const [ix,iy]=s2i(sx,sy);
    $("#posinfo").textContent = `x ${ix.toFixed(1)}  y ${iy.toFixed(1)}`;
    if(!drag || drag.mode!=="pan") drawLoupe(sx,sy);
  }
  if(!drag) return;
  if(drag.mode==="pan"){ ox=drag.ox0+(sx-drag.sx0); oy=drag.oy0+(sy-drag.sy0); draw(); return; }
  const [ix,iy]=s2i(sx,sy);
  if(drag.mode==="draw"){ const b=boxes[drag.k]; b.x2=ix; b.y2=iy; draw(); markDirty(); }
  else if(drag.mode==="move"){
    const b=boxes[drag.k], o=drag.box0, dx=ix-drag.ix0, dy=iy-drag.iy0;
    b.x1=o.x1+dx; b.y1=o.y1+dy; b.x2=o.x2+dx; b.y2=o.y2+dy; draw(); markDirty();
  } else if(drag.mode==="resize"){
    // non-dragged edges come from the fixed anchor box0, so dragging a handle
    // past the opposite edge pivots the box instead of collapsing it.
    const b=boxes[drag.k], o=drag.box0, h=drag.handle;
    b.x1 = h.includes("w") ? ix : o.x1;
    b.x2 = h.includes("e") ? ix : o.x2;
    b.y1 = h.includes("n") ? iy : o.y1;
    b.y2 = h.includes("s") ? iy : o.y2;
    draw(); markDirty();
  }
});

window.addEventListener("mouseup", ()=>{
  if(drag && (drag.mode==="draw")){
    const b=norm(boxes[drag.k]);
    if((b.x2-b.x1)<0.5 && (b.y2-b.y1)<0.5){ // click w/o drag -> tiny default box
      const cx=(b.x1+b.x2)/2, cy=(b.y1+b.y2)/2;
      boxes[drag.k]={x1:cx-3,y1:cy-3,x2:cx+3,y2:cy+3,label:b.label};
    } else { boxes[drag.k]=b; }
    draw(); markDirty();
  } else if(drag && (drag.mode==="move"||drag.mode==="resize")){
    boxes[drag.k]=norm(boxes[drag.k]); markDirty();
  }
  drag=null; syncUI();
});

view.addEventListener("wheel", e=>{
  e.preventDefault();
  zoomAt(e.offsetX, e.offsetY, e.deltaY<0 ? 1.15 : 1/1.15);
}, {passive:false});

// ---- keyboard ----
let spaceDown=false;
window.addEventListener("keydown", e=>{
  if(e.target.tagName==="INPUT") return;
  const k=e.key;
  if(k===" "){ spaceDown=true; view.style.cursor="grab"; e.preventDefault(); return; }
  if(k===","||k==="a"){ loadFrame(cur-1); e.preventDefault(); return; }
  if(k==="."||k==="d"){ loadFrame(cur+1); e.preventDefault(); return; }
  if(k==="PageDown"){ loadFrame(cur-10); return; }
  if(k==="PageUp"){ loadFrame(cur+10); return; }
  if(k==="c"){ copyPrev(); return; }
  if(k==="f"){ fitView(); return; }
  if(k==="0"){ zoomAt(view.width/2,view.height/2, 1/scale); return; }
  if(k==="+"||k==="="){ zoomAt(view.width/2,view.height/2,1.2); return; }
  if(k==="-"){ zoomAt(view.width/2,view.height/2,1/1.2); return; }
  if(k==="Escape"){ selected=-1; draw(); syncUI(); return; }
  if(k==="Delete"||k==="Backspace"){ if(selected>=0){ boxes.splice(selected,1); selected=-1; draw(); markDirty(); syncUI(); } e.preventDefault(); return; }
  if(/^[1-9]$/.test(k)){
    const idx=parseInt(k)-1;
    if(idx<CLASSES.length){
      activeClass=idx;
      if(selected>=0){ boxes[selected].label=CLASSES[idx]; draw(); markDirty(); }
      syncUI();
    }
    return;
  }
  // arrows: nudge selected box, else navigate
  if(k.startsWith("Arrow")){
    e.preventDefault();
    if(selected>=0){
      const step = e.shiftKey?5:(e.altKey?0.25:1);
      const b=boxes[selected];
      if(k==="ArrowLeft"){ b.x1-=step; b.x2-=step; }
      if(k==="ArrowRight"){ b.x1+=step; b.x2+=step; }
      if(k==="ArrowUp"){ b.y1-=step; b.y2-=step; }
      if(k==="ArrowDown"){ b.y1+=step; b.y2+=step; }
      draw(); markDirty();
    } else {
      if(k==="ArrowLeft") loadFrame(cur-1);
      if(k==="ArrowRight") loadFrame(cur+1);
    }
  }
});
window.addEventListener("keyup", e=>{ if(e.key===" "){ spaceDown=false; view.style.cursor="crosshair"; } });

// ---- actions ----
function copyPrev(){
  if(cur<=0) return;
  const prev = LABELS[cur-1] || [];
  boxes = boxes.concat(prev.map(b=>({...b})));
  draw(); markDirty(); syncUI();
}
$("#copyprev").onclick=copyPrev;
$("#clear").onclick=()=>{ boxes=[]; selected=-1; draw(); markDirty(); syncUI(); };
$("#prev").onclick=()=>loadFrame(cur-1);
$("#next").onclick=()=>loadFrame(cur+1);
$("#fit").onclick=fitView;
$("#one").onclick=()=>zoomAt(view.width/2,view.height/2,1/scale);
$("#zoomin").onclick=()=>zoomAt(view.width/2,view.height/2,1.2);
$("#zoomout").onclick=()=>zoomAt(view.width/2,view.height/2,1/1.2);
$("#carry").onclick=()=>{ carry=!carry; $("#carry").classList.toggle("on",carry); };
// blur after committing so global keyboard shortcuts resume (inputs swallow them)
$("#frameno").onchange=e=>{ loadFrame(parseInt(e.target.value)||0); e.target.blur(); };
$("#scrub").oninput=e=>loadFrame(parseInt(e.target.value));
$("#scrub").onchange=e=>e.target.blur();
$("#timeline").onclick=e=>{ const r=e.currentTarget.getBoundingClientRect();
  loadFrame(Math.round((e.clientX-r.left)/r.width*(INFO.n-1))); };
$("#export").onclick=async ()=>{
  const path = prompt("export GT to:", "work/gt_labeled.json");
  if(!path) return;
  await flush();                       // ensure the latest edit is persisted first
  // wait for the save queue to fully drain so the export sees every edit
  for(let i=0; (pending.size||flushing) && i<100; i++) await new Promise(r=>setTimeout(r,50));
  const r = await fetch("/api/export_gt", {method:"POST", body:JSON.stringify({path})});
  const j = await r.json();
  alert(j.ok ? `exported to ${j.path}` : "export failed");
};

// ---- persistence ----------------------------------------------------------
// Working edits live in `boxes`; committing writes them into LABELS[cur] and
// marks that frame `pending`. saveNow() drains `pending` one frame at a time,
// retrying on failure -- so a lost/failed PUT (or a frame we navigated away
// from) is never dropped. Each pending frame is keyed by its own index, so a
// retry always re-sends the RIGHT frame's boxes, not the current view's.
let dirty=false, saveTimer=null, flushing=false;
const pending=new Set();
function setSaved(txt, cls){ const el=$("#saved"); el.textContent=txt; el.className="pill "+cls; }
function cleanBoxes(){
  return boxes.map(b=>{ const n=norm(b);
    return {x1:+n.x1.toFixed(2),y1:+n.y1.toFixed(2),x2:+n.x2.toFixed(2),y2:+n.y2.toFixed(2),label:n.label}; })
    .filter(b => (b.x2-b.x1)>=0.5 || (b.y2-b.y1)>=0.5);
}
function commitLocal(){
  const clean=cleanBoxes();
  if(clean.length) LABELS[cur]=clean; else delete LABELS[cur];
  pending.add(cur); dirty=false;
}
function markDirty(){
  dirty=true; setSaved("unsaved","dirty");
  clearTimeout(saveTimer); saveTimer=setTimeout(flush, 500);
}
function flush(){                        // commit current frame, then push all pending
  clearTimeout(saveTimer);
  if(dirty) commitLocal();
  return saveNow();
}
function saveNow(){
  if(flushing) return Promise.resolve();
  if(pending.size===0){ if(!dirty) setSaved("saved","ok"); return Promise.resolve(); }
  flushing=true; setSaved("saving…","dirty");
  const f=pending.values().next().value;
  const payload=LABELS[f]||[];
  return fetch(`/api/labels/${f}`, {method:"PUT", body:JSON.stringify(payload)})
    .then(r=>{ if(!r.ok) throw new Error("http "+r.status); pending.delete(f); })
    .then(()=>{ flushing=false; renderTimeline();
                if(pending.size||dirty) return saveNow(); setSaved("saved","ok"); })
    .catch(()=>{ flushing=false; setSaved("save error — retrying","dirty");
                 clearTimeout(saveTimer); saveTimer=setTimeout(saveNow, 1500); });
}
window.addEventListener("beforeunload", ()=>{
  if(dirty) commitLocal();
  // sendBeacon can only POST; the backend accepts POST on /api/labels/ too.
  for(const f of pending) navigator.sendBeacon(`/api/labels/${f}`, JSON.stringify(LABELS[f]||[]));
});

// ---- UI sync ----
function renderClassbar(){
  const bar=$("#classbar"); bar.innerHTML="";
  CLASSES.forEach((c,i)=>{
    const row=document.createElement("div");
    row.className="cls"+(i===activeClass?" active":"");
    row.innerHTML=`<span class="k">${i+1}</span><span class="swatch" style="background:${PALETTE[i%PALETTE.length]}"></span><span>${c}</span>`;
    row.onclick=()=>{ activeClass=i; if(selected>=0){boxes[selected].label=c;draw();markDirty();} renderClassbar(); };
    bar.appendChild(row);
  });
  const add=document.createElement("div"); add.className="row";
  add.innerHTML=`<input id="newcls" type="text" placeholder="+ class" style="width:96px"><button id="addcls">add</button>`;
  bar.appendChild(add);
  const addCls=()=>{ const v=$("#newcls").value.trim(); if(v&&!CLASSES.includes(v)){
    CLASSES.push(v); activeClass=CLASSES.length-1;
    fetch("/api/classes",{method:"POST",body:JSON.stringify(CLASSES)});
    renderClassbar(); } view.focus(); };
  $("#addcls").onclick=addCls;
  $("#newcls").onkeydown=e=>{ if(e.key==="Enter"){ e.preventDefault(); addCls(); } };
}
function renderTimeline(){
  const tl=$("#timeline");
  tl.querySelectorAll(".tick").forEach(t=>t.remove());
  const W=tl.clientWidth||1;
  const step=Math.max(1, Math.floor(INFO.n/Math.min(INFO.n, W)));
  for(let i=0;i<INFO.n;i+=step){
    let any=false; for(let j=i;j<Math.min(INFO.n,i+step);j++) if(LABELS[j]&&LABELS[j].length){any=true;break;}
    if(any){ const t=document.createElement("div"); t.className="tick";
      t.style.left=(i/(INFO.n-1)*100)+"%"; tl.appendChild(t); }
  }
}
function syncUI(){
  $("#frameno").value=cur; $("#framecnt").textContent=`/ ${INFO.n-1}`;
  $("#scrub").value=cur; $("#cursorline").style.left=(cur/(INFO.n-1)*100)+"%";
  $("#zoomlbl").textContent=Math.round(scale*100)+"%";
  $("#boxinfo").textContent=`${boxes.length} box${boxes.length===1?"":"es"}`+(selected>=0?" · 1 sel":"");
  renderClassbar();
}

// ---- boot ----
async function boot(){
  const r = await fetch("/api/info"); const j = await r.json();
  INFO = j.info; CLASSES = j.classes.slice(); LABELS = {};
  for(const [k,v] of Object.entries(j.labels)) LABELS[parseInt(k)] = v;
  $("#frameno").max=INFO.n-1; $("#scrub").max=INFO.n-1;
  $("#framecnt").textContent=`/ ${INFO.n-1}`;
  document.title = `label · ${INFO.video}`;
  resize(); fitView();
  loadFrame(0, {commit:false});
  renderTimeline();
  window.addEventListener("resize", resize);
}
boot();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(video: str, labels_path: str, from_gt: str | None,
        classes: list[str] | None, host: str, port: int, no_open: bool) -> None:
    video = str(_resolve(video))
    from_gt = str(_resolve(from_gt)) if from_gt else None
    if not Path(video).exists():
        raise SystemExit(f"video not found: {video}\n"
                         f"(pass --video with a path relative to {ROOT} or an absolute path)")
    jpegs = load_frames(video)
    info = probe(video)

    lpath = _resolve(labels_path)
    if lpath.exists():
        labels = json.loads(lpath.read_text())
        labels.setdefault("frames", {})
        labels.setdefault("classes", classes or DEFAULT_CLASSES)
        print(f"loaded {len(labels['frames'])} labeled frames from {lpath}")
    else:
        seed_frames: dict[str, list[dict]] = {}
        seed_classes: list[str] = []
        if from_gt:
            seed_frames, seed_classes = gt_to_labels(from_gt)
            print(f"seeded {len(seed_frames)} frames from {from_gt}")
        labels = {
            "video": Path(video).name,
            "width": info.width,
            "height": info.height,
            "classes": classes or (seed_classes or DEFAULT_CLASSES),
            "frames": seed_frames,
        }

    STATE["jpegs"] = jpegs
    STATE["info"] = {"video": Path(video).name, "n": len(jpegs),
                     "width": info.width, "height": info.height}
    STATE["labels"] = labels
    STATE["labels_path"] = lpath
    with STATE["lock"]:
        save_labels()

    # bind, retrying a few ports if busy
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {port}..{port+20}")

    url = f"http://{host}:{port}/"
    print(f"\n  labeler ready -> {url}")
    print(f"  labels: {lpath}   ({len(jpegs)} frames)\n  Ctrl-C to stop\n", flush=True)
    if not no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        server.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description="frame-by-frame bbox labeler")
    ap.add_argument("--video", default="07_05.mp4")
    ap.add_argument("--labels", default="work/labels.json",
                    help="per-frame label store (read + written)")
    ap.add_argument("--from-gt", help="seed an empty label store from a GT json")
    ap.add_argument("--classes", help="comma-separated class names")
    ap.add_argument("--export-gt", metavar="PATH",
                    help="convert the label store to GT format and exit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    a = ap.parse_args()

    if a.export_gt:
        labels = json.loads(_resolve(a.labels).read_text())
        labels_to_gt(labels, _resolve(a.export_gt))
        return

    classes = [c.strip() for c in a.classes.split(",")] if a.classes else None
    run(a.video, a.labels, a.from_gt, classes, a.host, a.port, a.no_open)


if __name__ == "__main__":
    main()
