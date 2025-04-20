'''
Omair Ahmad - 251443603
Yukti. - 251400558
------------------------
AI-Powered Real-Time Suspicious Behavior Detection in CCTV Footage
------------------------
This code implements weapon detection which is a part of the System Architecture's, Weapon Detection Module.
'''

import os, cv2, numpy as np
from ultralytics import YOLO

model = YOLO("weapons_trained_yolo11n.pt")

image_path = "test images/test1.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 640))

results = model.predict(image)
annotated_frame = results[0].plot()

output_dir = "/Users/omairahmad_/Desktop/BAI - Project/final/WEAPON DETECTION/outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "annotated_test1.jpg")

cv2.imwrite(output_path, annotated_frame)
print(f"Saved annotated image to: {output_path}")