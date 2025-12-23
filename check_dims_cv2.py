
import cv2
import sys

path = r'c:\Users\worap\OneDrive\Desktop\Final_Project\photobooth_backend\frames\Frame.png'
img = cv2.imread(path)
if img is not None:
    h, w = img.shape[:2]
    print(f"DIMENSIONS: {w}x{h}")
    
    # Check aspect ratio 2:3 (or 3:2)
    ratio = w / h
    target_ratio = 2/3
    inverse_target = 3/2
    
    print(f"RATIO: {ratio:.2f}")
    if 0.65 < ratio < 0.68 or 1.48 < ratio < 1.52:
        print("STATUS: GOOD Aspect Ratio for 4x6")
    else:
        print("STATUS: BAD Aspect Ratio for 4x6")
else:
    print("Failed to load image")
