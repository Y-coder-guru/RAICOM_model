from ultralytics import YOLO

# 1. 加载你训练好的最优模型
model = YOLO("runs/detect/train2/weights/best.pt")

# 2. 重新跑验证（会自动生成所有缺失图）
model.val(data="D:\\y_rtxq\\RAICOM_yolov8\\datasets\\data.yaml", plots=True)