import os
import glob
import re
import uuid
import time
import sys
import shutil
import subprocess
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
TEMP_DIR = os.path.join(BASE_DIR, "temp_shots")
UPLOAD_DIR = os.path.join(BASE_DIR, "captured_photos")

DIGICAM_CMD_PATH = os.getenv(
    "DIGICAM_CMD_PATH",
    r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe",
)

for d in (FRAMES_DIR, TEMP_DIR, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

allow_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
origins = ["*"] if allow_origins.strip() == "*" else [o.strip() for o in allow_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to ensure CORS headers on all responses (including static files)
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")
app.mount("/temp_view", StaticFiles(directory=TEMP_DIR), name="temp_view")


def fit_center_crop(img, target_w, target_h):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None
    scale = max(target_w / w, target_h / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    rh, rw = resized.shape[:2]
    x = max(0, (rw - target_w) // 2)
    y = max(0, (rh - target_h) // 2)
    cropped = resized[y : y + target_h, x : x + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(cropped, (target_w, target_h))
    return cropped


def detect_slots_and_mask(frame_path):
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return [], None, None

    fh, fw = image.shape[:2]
    mask = None

    name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    mask_path = os.path.join(os.path.dirname(frame_path), f"{name_no_ext}_mask.png")

    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        if len(image.shape) == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.min(alpha) < 255:
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY_INV)

        if mask is None:
            if len(image.shape) == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    if mask is None:
        return [], None, image

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.05 and h > fh * 0.05:
            slots.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    slots.sort(key=lambda s: (s["y"] // 20, s["x"]))
    return slots, mask, image


def fix_frame_aspect_ratio(frame_path: str, target_width: int = 1200, target_height: int = 1800):
    if not os.path.exists(frame_path):
        return {"error": "File not found"}

    img = cv2.imread(frame_path)
    if img is None:
        return {"error": "Failed to load image"}

    h, w = img.shape[:2]
    
    # Check if already correct
    if w == target_width and h == target_height:
        return {"status": "ok", "message": "Already correct dimensions"}

    # Backup
    backup_path = frame_path + ".bak"
    shutil.copy2(frame_path, backup_path)

    # Resize with Aspect Fill (Center Crop)
    aspect_ratio = w / h
    target_ratio = target_width / target_height 
    
    if aspect_ratio > target_ratio:
        # Image is wider than target -> Fix Height, Crop Width
        new_h = target_height
        new_w = int(w * (target_height / h))
    else:
        # Image is taller/equal -> Fix Width, Crop Height
        new_w = target_width
        new_h = int(h * (target_width / w))
        
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center Crop
    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    
    cropped = resized[start_y:start_y+target_height, start_x:start_x+target_width]
    
    cv2.imwrite(frame_path, cropped)
    return {"status": "fixed", "original": f"{w}x{h}", "new": f"{target_width}x{target_height}"}



def base_url_from_request(request: Request) -> str:
    forced = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if forced:
        return forced
    return str(request.base_url).rstrip("/")


@app.get("/")
def root():
    return {"status": "ok", "service": "photobooth-backend"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/frames-list")
def get_frames_list(request: Request):
    base = base_url_from_request(request)
    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for f in glob.glob(os.path.join(FRAMES_DIR, ext)):
            name = os.path.basename(f)
            if "_mask" in name:
                continue
            frames.append({"id": name, "name": name, "url": f"{base}/frames/{name}"})
    frames.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x["name"])])
    return frames


@app.get("/frame-props/{frame_id}")
def get_frame_props(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    if not os.path.exists(frame_path):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    slots_data, _, img = detect_slots_and_mask(frame_path)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid Image"})
    fh, fw = img.shape[:2]
    norm_slots = [{"x": s["x"] / fw, "y": s["y"] / fh, "w": s["w"] / fw, "h": s["h"] / fh} for s in slots_data]
    return {"width": fw, "height": fh, "slots": norm_slots or []}


@app.post("/fix-frame/{frame_id}")
def api_fix_frame(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    result = fix_frame_aspect_ratio(frame_path)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result



@app.post("/capture_step")
async def upload_capture_step(step: int = Form(...), file: UploadFile = File(...)):
    if step < 1 or step > 10:
        return JSONResponse(status_code=400, content={"error": "Invalid step"})
    out_path = os.path.join(TEMP_DIR, f"temp_{step}.jpg")
    try:
        with open(out_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"status": "success", "saved_as": f"temp_{step}.jpg"}


@app.post("/trigger_dslr")
def trigger_dslr(request: Request, step: int = Form(...)):
    if step < 1 or step > 10:
        return JSONResponse(status_code=400, content={"error": "Invalid step"})

    if sys.platform != "win32":
        return JSONResponse(status_code=400, content={"error": "DSLR trigger is supported on Windows only"})

    if not os.path.exists(DIGICAM_CMD_PATH):
        return JSONResponse(status_code=500, content={"error": "DIGICAM_CMD_PATH not found", "path": DIGICAM_CMD_PATH})

    filename = f"temp_{step}.jpg"
    save_path = os.path.join(TEMP_DIR, filename)

    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except Exception:
            pass

    cmd = [DIGICAM_CMD_PATH, "/capture", "/filename", save_path]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if not os.path.exists(save_path):
            return JSONResponse(status_code=500, content={"error": "Capture failed (File not found)"})
        base = base_url_from_request(request)
        return {"status": "success", "mode": "dslr", "image_url": f"{base}/temp_view/{filename}?t={time.time()}"}
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=500, content={"error": "Capture timeout"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/merge")
def merge_photos(request: Request, frame_id: str = Form(...)):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    if not os.path.exists(frame_path):
        return JSONResponse(status_code=404, content={"error": "Frame not found"})

    slots_data, mask, frame = detect_slots_and_mask(frame_path)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid frame image"})
    if not slots_data:
        return JSONResponse(status_code=400, content={"error": "No slots detected for this frame"})

    canvas = frame.copy()
    if len(canvas.shape) == 3 and canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    elif len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGRA)

    for i, s in enumerate(slots_data):
        shot_path = os.path.join(TEMP_DIR, f"temp_{i+1}.jpg")
        if not os.path.exists(shot_path):
            continue
        shot = cv2.imread(shot_path)
        if shot is None:
            continue

        fitted = fit_center_crop(shot, s["w"], s["h"])
        if fitted is None:
            continue

        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        if y + h > canvas.shape[0] or x + w > canvas.shape[1]:
            continue

        roi = canvas[y : y + h, x : x + w]

        if mask is not None:
            mask_roi = mask[y : y + h, x : x + w]
            if mask_roi is not None and mask_roi.shape[0] == h and mask_roi.shape[1] == w:
                alpha = (mask_roi.astype(np.float32) / 255.0)[..., None]
                base_rgb = roi[:, :, :3].astype(np.float32)
                fitted_rgb = fitted[:, :, :3].astype(np.float32)
                blended = fitted_rgb * alpha + base_rgb * (1.0 - alpha)
                roi[:, :, :3] = blended.astype(np.uint8)
            else:
                roi[:, :, :3] = fitted[:, :, :3]
        else:
            roi[:, :, :3] = fitted[:, :, :3]

    final_name = f"{uuid.uuid4()}.jpg"
    out_path = os.path.join(UPLOAD_DIR, final_name)
    cv2.imwrite(out_path, canvas[:, :, :3])

    base = base_url_from_request(request)
    return {"status": "success", "image_url": f"{base}/photos/{final_name}?t={time.time()}", "filename": final_name}



@app.post("/print/{filename}")
def print_photo(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
    
    try:
        if sys.platform == "win32":
            # User requested the standard Windows Print dialog
            os.startfile(file_path, "print")
        else:
            subprocess.run(["lpr", file_path], check=False)
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})



@app.delete("/cleanup")
def cleanup():
    for d in (TEMP_DIR, UPLOAD_DIR):
        for f in glob.glob(os.path.join(d, "*")):
            try:
                os.remove(f)
            except Exception:
                pass
    return {"status": "cleaned"}

@app.post("/print_upload")
async def print_upload(file: UploadFile = File(...)):
    """
    Receives a blob/file from frontend and prints it using system default printer or specific command.
    """
    # Create a unique filename
    filename = f"print_{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Print logic
        if sys.platform == "win32":
            # Use 'print' verb to trigger silent print if default printer is set up to not ask dialogs
            # OR better: use mspaint /p or a specific command line tool like IrfanView if available for true silent print.
            # os.startfile(file_path, "print") opens the dialog in some viewers, prints directly in others.
            # For true silent print on Windows without extra tools, powershell or rundll32 might be needed, 
            # but os.startfile("print") is the standard "Ask OS to print this".
            os.startfile(file_path, "print") 
        else:
            subprocess.run(["lpr", file_path], check=False)
            
        return {"status": "success", "message": "Sent to printer"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
