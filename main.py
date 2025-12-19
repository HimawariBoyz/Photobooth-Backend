import cv2
import os
import glob
import uuid
import numpy as np
import shutil
import uvicorn
import sys
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI()

# 1. Config CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
TEMP_DIR = os.path.join(BASE_DIR, "temp_shots")
UPLOAD_DIR = os.path.join(BASE_DIR, "captured_photos")

for d in [FRAMES_DIR, TEMP_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. Mount Static Files
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")

# --- Helper Functions (Smart CV) ---

def resize_cover(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    x = (nw - target_w) // 2
    y = (nh - target_h) // 2
    return resized[y:y+target_h, x:x+target_w]

def detect_slots_from_image(frame_path):
    """
    ระบบค้นหาช่องว่างอัตโนมัติ (รองรับทั้งแบบมี Mask และไม่มี)
    """
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None: return [], None, None

    fh, fw = image.shape[:2]
    mask = None

    # 1. ลองหาจากไฟล์ Mask แยกก่อน (ถ้ามี)
    name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    mask_path_png = os.path.join(os.path.dirname(frame_path), f"{name_no_ext}_mask.png")
    
    if os.path.exists(mask_path_png):
        mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
    else:
        # 2. ถ้าไม่มีไฟล์ Mask ให้ Auto Detect จากความโปร่งใส หรือ สีขาว
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.min(alpha) < 255:
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY_INV)
        
        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # หาเส้นขอบ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.05 and h > fh * 0.05: # กรอง Noise
            slots.append({"x": x, "y": y, "w": w, "h": h})

    # เรียงลำดับช่อง
    slots.sort(key=lambda s: (s['y'] // 20, s['x']))
    return slots, mask, image

# --- API Routes ---

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
    if not os.path.exists(frame_path):
        return JSONResponse(status_code=404, content={"error": "Frame not found"})

    slots_data, _, img = detect_slots_from_image(frame_path)
    if img is None: return {"error": "Invalid Image"}
    
    fh, fw = img.shape[:2]
    # แปลงเป็น % ส่งให้ Frontend
    normalized_slots = [{"x": s['x']/fw, "y": s['y']/fh, "w": s['w']/fw, "h": s['h']/fh} for s in slots_data]
    
    # Fallback Default
    if not normalized_slots:
         normalized_slots = [
            {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.35},
            {"x": 0.55, "y": 0.1, "w": 0.35, "h": 0.35},
            {"x": 0.1, "y": 0.55, "w": 0.35, "h": 0.35},
            {"x": 0.55, "y": 0.55, "w": 0.35, "h": 0.35},
        ]
    return {"width": fw, "height": fh, "slots": normalized_slots}

@app.post("/capture_step")
async def upload_capture_step(step: int = Form(...), file: UploadFile = File(...)):
    file_location = os.path.join(TEMP_DIR, f"temp_{step}.jpg")
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "step": step}

@app.post("/merge")
def merge_photos(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    if not os.path.exists(frame_path):
        return JSONResponse(status_code=404, content={"message": "Frame not found"})

    # ใช้ Smart CV Detect Slots เพื่อความแม่นยำ
    slots_data, mask, frame = detect_slots_from_image(frame_path)
    
    # Prepare Canvas
    canvas = frame.copy()
    if canvas.shape[2] == 4:
        # แปลงเป็นพื้นขาวถ้าต้นฉบับมี Alpha
        bg = np.ones_like(canvas[:,:,:3]) * 255
        alpha = canvas[:,:,3] / 255.0
        for c in range(3):
            bg[:,:,c] = canvas[:,:,c] * alpha + bg[:,:,c] * (1 - alpha)
        canvas = bg.astype(np.uint8)

    # Place Photos
    for i, s in enumerate(slots_data):
        shot_path = os.path.join(TEMP_DIR, f"temp_{i+1}.jpg")
        if not os.path.exists(shot_path): continue

        shot = cv2.imread(shot_path)
        x, y, w, h = s['x'], s['y'], s['w'], s['h']
        fitted = resize_cover(shot, w, h)
        canvas[y:y+h, x:x+w] = fitted

    # Save High-Res File
    final_name = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, final_name), canvas)

    return {
        "status": "success", 
        "image_url": f"http://localhost:8000/photos/{final_name}",
        "filename": final_name
    }

# --- NEW: Print API ---
@app.post("/print/{filename}")
def print_photo(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    
    try:
        if sys.platform == "win32":
            # Windows: สั่งปริ้นผ่าน Default Printer
            os.startfile(file_path, "print")
        else:
            # Mac/Linux: ใช้คำสั่ง lp หรือ lpr
            subprocess.run(["lpr", file_path])
            
        return {"status": "success", "message": "Sent to printer"}
    except Exception as e:
        print("Print Error:", e)
        return {"status": "error", "message": str(e)}

@app.delete("/cleanup")
def cleanup():
    for d in [TEMP_DIR, UPLOAD_DIR]:
        for f in glob.glob(os.path.join(d, "*")):
            try: os.remove(f)
            except: pass
    return {"status": "cleaned"}

if __name__ == "__main__":
    print("🚀 Photobooth Server Running...")
    uvicorn.run(app, host="0.0.0.0", port=8000)