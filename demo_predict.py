'''
from ultralytics import YOLO
import cv2

yolo = YOLO("./best.pt",task = "detect")

result = yolo(source=r"D:\辉探智控\RAICOM_yolov8\datasets\train\images\18.jpg",show=True)
#0表示电脑摄像头、screen屏幕
#yolo detect predict model=runs/detect/train/weights/best.pt source=./BVN.mp4 show=True

# 关键：无限等待按键，按任意键关闭窗口
cv2.waitKey(0)
# 释放窗口资源
cv2.destroyAllWindows()
'''


from ultralytics import YOLO
import cv2

model = YOLO(r"D:\y_rtxq\RAICOM_yolov8\runs\detect\train2\weights\best.pt", task="detect")
results = model(source=r"D:\y_rtxq\RAICOM_yolov8\datasets\train\images\30_aug2.jpg")

result = results[0]
annotated_img = result.plot() 

scale_percent = 50              #缩放到原来的50%
width = int(annotated_img.shape[1] * scale_percent / 100)
height = int(annotated_img.shape[0] * scale_percent / 100)
annotated_img = cv2.resize(annotated_img, (width, height))

cv2.imshow("Detection Result", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()