
import cv2
import shutil
import os

FRAME_PATH = r'frames/Frame.png'
TARGET_WIDTH = 1200
TARGET_HEIGHT = 1800

def fix_frame():
    if not os.path.exists(FRAME_PATH):
        print(f"Error: {FRAME_PATH} not found.")
        return

    img = cv2.imread(FRAME_PATH)
    if img is None:
        print("Error: Failed to load image.")
        return

    h, w = img.shape[:2]
    print(f"Original Size: {w}x{h}")
    
    # Target 4x6 aspect ratio is 2:3
    # 1200x1800 is standard 300dpi 4x6
    
    if w == TARGET_WIDTH and h == TARGET_HEIGHT:
        print("Frame is already 1200x1800 (4x6). No changes needed.")
        return

    # Backup
    backup_path = FRAME_PATH + ".bak"
    shutil.copy2(FRAME_PATH, backup_path)
    print(f"Backed up original to {backup_path}")

    # Resize with Aspect Fill (Center Crop)
    aspect_ratio = w / h
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT # 0.666
    
    # Strategy: Resize so that the image COVERS the target area, then crop center
    
    if aspect_ratio > target_ratio:
        # Image is wider than target (e.g. 16:9 vs 2:3)
        # Fix Height, Crop Width
        new_h = TARGET_HEIGHT
        new_w = int(w * (TARGET_HEIGHT / h))
    else:
        # Image is taller than target (or equal width)
        # Fix Width, Crop Height
        new_w = TARGET_WIDTH
        new_h = int(h * (TARGET_WIDTH / w))
        
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center Crop
    start_x = (new_w - TARGET_WIDTH) // 2
    start_y = (new_h - TARGET_HEIGHT) // 2
    
    cropped = resized[start_y:start_y+TARGET_HEIGHT, start_x:start_x+TARGET_WIDTH]
    
    print(f"Resized to {new_w}x{new_h}, Cropped to {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    cv2.imwrite(FRAME_PATH, cropped)
    print(f"Saved fixed frame to {FRAME_PATH}")

if __name__ == "__main__":
    fix_frame()
