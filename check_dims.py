from PIL import Image
import os

path = r'c:\Users\worap\OneDrive\Desktop\Final_Project\photobooth_backend\frames\Frame.png'
if os.path.exists(path):
    img = Image.open(path)
    print(f"SIZE: {img.size}")
    print(f"MODE: {img.mode}")
else:
    print("File not found")
