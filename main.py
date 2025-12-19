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

# --- 1. Config & Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
TEMP_DIR = os.path.join(BASE_DIR, "temp_shots")
UPLOAD_DIR = os.path.join(BASE_DIR, "captured_photos")

# Path โปรแกรม digiCamControl (เช็คเครื่องคุณอีกทีนะครับ)
DIGICAM_CMD_PATH = r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe"

for d in [FRAMES_DIR, TEMP_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# 2. Config CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount Static Files
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")
app.mount("/temp_view", StaticFiles(directory=TEMP_DIR), name="temp_view")

# --- Helper Functions (รวมของคุณ + ของเก่า) ---

def resize_cover(img, target_w, target_h):
    """ ปรับขนาดรูปให้เต็มช่อง (Cover) """
    if img is None: return None
    h, w = img.shape[:2]
    if w == 0 or h == 0: return img
    
    scale = max(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))

    x = (nw - target_w) // 2
    y = (nh - target_h) // 2
    return resized[y:y+target_h, x:x+target_w]

def detect_slots_and_mask(frame_path):
    """ 
    ค้นหาช่องว่างและ Mask:
    - ถ้ามีไฟล์ _mask.png จะใช้ไฟล์นั้น
    - ถ้าไม่มี จะ Auto detect จากความโปร่งใส (Alpha)
    """
    image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if image is None: return [], None, None

    fh, fw = image.shape[:2]
    mask = None

    # 1. หาไฟล์ Mask แยก (priority สูงสุด)
    name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    mask_path_png = os.path.join(os.path.dirname(frame_path), f"{name_no_ext}_mask.png")
    
    if os.path.exists(mask_path_png):
        mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
    else:
        # 2. ถ้าไม่มี Mask ให้สร้างเองจาก Alpha Channel
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            # ส่วนที่ใส (Alpha < 255) คือช่องว่าง
            if np.min(alpha) < 255:
                # Invert: ให้ช่องว่างเป็นสีขาว (255) เพื่อหา Contour
                _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Fallback: ถ้าเป็น JPG หรือไม่มี Alpha ให้เดาจากสีขาว/ดำ
        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # หา Contours (ช่องใส่รูป)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    slots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > fw * 0.05 and h > fh * 0.05: # กรองจุดรบกวนเล็กๆ
            slots.append({"x": x, "y": y, "w": w, "h": h, "cnt": cnt}) # เก็บ contour ไว้ใช้ตอน merge

    # เรียงลำดับช่อง: บนลงล่าง, ซ้ายไปขวา
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

    slots_data, _, img = detect_slots_and_mask(frame_path)
    if img is None: return {"error": "Invalid Image"}
    
    fh, fw = img.shape[:2]
    # ส่งพิกัดเป็น % ให้ Frontend
    normalized_slots = [{"x": s['x']/fw, "y": s['y']/fh, "w": s['w']/fw, "h": s['h']/fh} for s in slots_data]
    
    if not normalized_slots:
        # ค่า Default กันตาย
        normalized_slots = [
            {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.35},
            {"x": 0.55, "y": 0.1, "w": 0.35, "h": 0.35},
            {"x": 0.1, "y": 0.55, "w": 0.35, "h": 0.35},
            {"x": 0.55, "y": 0.55, "w": 0.35, "h": 0.35},
        ]
    return {"width": fw, "height": fh, "slots": normalized_slots}

# --- 1. Webcam Upload ---
@app.post("/capture_step")
async def upload_capture_step(step: int = Form(...), file: UploadFile = File(...)):
    file_location = os.path.join(TEMP_DIR, f"temp_{step}.jpg")
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success", "mode": "webcam", "step": step}

# --- 2. DSLR Trigger ---
@app.post("/trigger_dslr")
def trigger_dslr(step: int = Form(...)):
    filename = f"temp_{step}.jpg"
    save_path = os.path.join(TEMP_DIR, filename)
    
    if os.path.exists(save_path):
        try: os.remove(save_path)
        except: pass

    cmd = [DIGICAM_CMD_PATH, "/capture", "/filename", save_path]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if os.path.exists(save_path):
            return {
                "status": "success", 
                "mode": "dslr",
                "image_url": f"http://localhost:8000/temp_view/{filename}?t={time.time()}"
            }
        else:
            return JSONResponse(status_code=500, content={"error": "Capture failed"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 🔥 MERGE LOGIC (อัปเดตใช้ Alpha Blending ตามโค้ดที่คุณให้มา) ---
@app.post("/merge")
def merge_photos(frame_id: str):
    frame_path = os.path.join(FRAMES_DIR, frame_id)
    if not os.path.exists(frame_path):
        return JSONResponse(status_code=404, content={"message": "Frame not found"})

    # 1. โหลดข้อมูล Slot และ Mask
    slots_data, mask, frame = detect_slots_and_mask(frame_path)
    
    # 2. เตรียม Canvas (Frame Original)
    canvas = frame.copy()
    
    # ถ้าภาพ Frame ไม่มี Alpha ให้ใส่ Alpha เป็นทึบไว้ก่อน (เพื่อให้คำนวณ Blending ได้)
    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

    # 3. วนลูปแปะรูปตาม Slot
    for i, s in enumerate(slots_data):
        shot_path = os.path.join(TEMP_DIR, f"temp_{i+1}.jpg")
        if not os.path.exists(shot_path): continue

        shot = cv2.imread(shot_path)
        if shot is None: continue

        x, y, w, h = s['x'], s['y'], s['w'], s['h']
        
        # ปรับขนาดรูปคนให้พอดีช่อง
        fitted = resize_cover(shot, w, h)
        
        # --- 🔥 ALPHA BLENDING LOGIC (จากโค้ดที่คุณให้มา) ---
        # ดึงพื้นที่ใน Canvas ที่จะวางรูป (ROI)
        roi = canvas[y:y+h, x:x+w]
        
        # ดึง Mask ตรงช่องนั้น (ทำให้เป็น 0.0 - 1.0)
        # ถ้า detect_slots_and_mask ส่ง mask มา
        if mask is not None:
             # ตัด Mask เฉพาะส่วน ROI
             mask_roi = mask[y:y+h, x:x+w]
             
             # แปลง Mask เป็น Alpha (0-1)
             # Mask สีขาว (255) คือช่องว่าง -> รูปคนต้องชัด (Alpha=1)
             # Mask สีดำ (0) คือเนื้อกรอบ -> รูปคนต้องจาง/หาย (Alpha=0)
             alpha_channel = mask_roi / 255.0
             alpha_channel = alpha_channel[..., None] # ขยายมิติให้คูณกับ BGR ได้

             # สูตร: (รูปคน * alpha) + (พื้นหลังเดิม * (1-alpha))
             # ถ้าใช้ Mask ที่ Invert มาแล้ว (ช่องว่าง=255) ให้ใช้สูตรนี้
             blended = (fitted * alpha_channel + roi[:,:,:3] * (1 - alpha_channel))
             
             # อัปเดต ROI (เอาแค่ RGB ไม่เอา Alpha ของตัว ROI เพราะเราจะทับเลย)
             roi[:,:,:3] = blended.astype(np.uint8)
        else:
             # กรณีไม่มี Mask เลย แปะทับดื้อๆ
             roi[:,:,:3] = fitted

    # 4. แปลงกลับเป็น JPG (ตัด Alpha ทิ้งก่อนเซฟ) เพื่อให้ไฟล์เล็ก
    final_img_rgb = canvas[:,:,:3]
    
    final_name = f"{uuid.uuid4()}.jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, final_name), final_img_rgb)

    return {
        "status": "success", 
        "image_url": f"http://localhost:8000/photos/{final_name}",
        "filename": final_name
    }

@app.post("/print/{filename}")
def print_photo(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    try:
        if sys.platform == "win32":
            os.startfile(file_path, "print")
        else:
            subprocess.run(["lpr", file_path])
        return {"status": "success", "message": "Sent to printer"}
    except Exception as e:
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