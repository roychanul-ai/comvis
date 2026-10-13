import torch
import cv2
import matplotlib
import ultralytics

print("PyTorch:", torch.__version__)
print("CUDA tersedia:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Ultralytics:", ultralytics.__version__)