from ultralytics import YOLO
import torch


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

model = YOLO('models/best.pt')

results = model.predict('SLIIT/sliitvssj.mp4', save=True)
print(results[0])
print('+++++++++++++++++++++++++++++++++')
for box in results[0].boxes:
    print(box)
