import cv2
import os
import glob
import uuid
import numpy as np
import shutil
import uvicorn
import sys
import subprocess
import time 
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
TEMP_DIR = os.path.join(BASE_DIR, "temp_shots")
UPLOAD_DIR = os.path.join(BASE_DIR, "captured_photos")

# ✅ Path ของ digiCamControl (ตัวกลางคุยกับ Sony)
DIGICAM_CMD_PATH = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe" 

for d in [FRAMES_DIR, TEMP_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")
app.mount("/temp_view", StaticFiles(directory=TEMP_DIR), name="temp_view")

# --- 🔥 THE SECRET SAUCE: fit_center_crop ---
def fit_center_crop(img, target_w, target_h):
    if img is None: return None
    h, w = img.shape[:2]
    
    # 1. Scale ให้ชนขอบ (Max Scale)
    scale = max(target_w / w, target_h / h)
    
    # 2. Resize
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # 3. Center Crop
    rh, rw = resized.shape[:2]
    x = (rw - target_w) // 2
    y = (rh - target_h) // 2
    
    return resized[y:y+target_h, x:x+target_w]

def detect_slots_and_mask(frame_path):
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None: return [], None, None
    fh, fw = image.shape[:2]
    mask = None
    
    # Find Mask
    name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    mask_path = os.path.join(os.path.dirname(frame_path), f"{name_no_ext}_mask.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Auto Mask from Alpha
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.min(alpha) < 255:
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY_INV)
        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find Slots
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.05 and h > fh * 0.05:
            slots.append({"x": x, "y": y, "w": w, "h": h}) 

    slots.sort(key=lambda s: (s['y'] // 20, s['x']))
    return slots, mask, image

# --- Routes ---

@app.get("/frames-list")
def get_frames_list():
    frames = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for f in glob.glob(os.path.join(FRAMES_DIR, ext)):
            name = os.path.basename(f)
            if "_mask" in name: continue 
            frames.append({
                "id": name,
                "name": name,
                "url": f"http://localhost:8000/frames/{name}"
            })
    return frames

@app.get("/frame-props/{frame_id}")
def get_frame_props(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    if not os.path.exists(frame_path): return {"error": "Not found"}
    slots_data, _, img = detect_slots_and_mask(frame_path)
    if img is None: return {"error": "Invalid Image"}
    fh, fw = img.shape[:2]
    norm_slots = [{"x": s['x']/fw, "y": s['y']/fh, "w": s['w']/fw, "h": s['h']/fh} for s in slots_data]
    return {"width": fw, "height": fh, "slots": norm_slots or []}

# --- 📸 1. รับรูปจาก Webcam (กรณีใช้โหมด Video/HDMI) ---
@app.post("/capture_step")
async def upload_capture_step(step: int = Form(...), file: UploadFile = File(...)):
    with open(os.path.join(TEMP_DIR, f"temp_{step}.jpg"), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success"}

# --- 📸 2. สั่งถ่ายจาก Sony ผ่าน USB (กรณีใช้โหมด PC Remote) ---
@app.post("/trigger_dslr")
def trigger_dslr(step: int = Form(...)):
    filename = f"temp_{step}.jpg"
    save_path = os.path.join(TEMP_DIR, filename)
    
    # ลบไฟล์เก่าทิ้งก่อน
    if os.path.exists(save_path):
        try: os.remove(save_path)
        except: pass

    # คำสั่งยิงชัตเตอร์ (ใช้ได้ทั้ง Nikon และ Sony ที่ต่อ PC Remote)
    cmd = [DIGICAM_CMD_PATH, "/capture", "/filename", save_path]
    
    try:
        # เพิ่ม Timeout เป็น 15 วินาที เผื่อ Sony โฟกัสช้า
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        if os.path.exists(save_path):
            return {
                "status": "success", 
                "mode": "dslr",
                # ใส่ ?t=time เพื่อแก้ Browser Cache
                "image_url": f"http://localhost:8000/temp_view/{filename}?t={time.time()}"
            }
        else:
            return JSONResponse(status_code=500, content={"error": "Capture failed (File not found)"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/merge")
def merge_photos(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    slots_data, mask, frame = detect_slots_and_mask(frame_path)
    
    canvas = frame.copy()
    if canvas.shape[2] == 3: canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

    for i, s in enumerate(slots_data):
        shot_path = os.path.join(TEMP_DIR, f"temp_{i+1}.jpg")
        if not os.path.exists(shot_path): continue
        shot = cv2.imread(shot_path)
        if shot is None: continue

        # 🔥 ใช้ fit_center_crop ตัดขอบดำออกอัตโนมัติ
        fitted = fit_center_crop(shot, s['w'], s['h'])
        
        x, y, w, h = s['x'], s['y'], s['w'], s['h']
        roi = canvas[y:y+h, x:x+w]
        
        if mask is not None:
             mask_roi = mask[y:y+h, x:x+w]
             alpha = (mask_roi / 255.0)[..., None]
             blended = (fitted * alpha + roi[:,:,:3] * (1 - alpha))
             roi[:,:,:3] = blended.astype(np.uint8)
        else:
             roi[:,:,:3] = fitted

    final_name = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, final_name), canvas[:,:,:3])
    return {"status": "success", "image_url": f"http://localhost:8000/photos/{final_name}", "filename": final_name}

@app.post("/print/{filename}")
def print_photo(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    try:
        if sys.platform == "win32": os.startfile(file_path, "print")
        else: subprocess.run(["lpr", file_path])
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.delete("/cleanup")
def cleanup():
    for d in [TEMP_DIR, UPLOAD_DIR]:
        for f in glob.glob(os.path.join(d, "*")):
            try: os.remove(f)
            except: pass
    return {"status": "cleaned"}

if __name__ == "__main__":
    print("🚀 Sony Photobooth Server Running...")
    uvicorn.run(app, host="0.0.0.0", port=8000)