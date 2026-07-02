'''
from ultralytics import YOLO
import os

# 解决OpenMP库重复加载警告
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 关闭YOLO联网检测,防超时
os.environ["ULTRALYTICS_OFFLINE"]="1"

model=YOLO(r".\yolov8n.pt")

#                                     win下一般0    总轮数     批次大小  学习率       优化器           权重衰减        余弦退火学习率   验证集    断点续训
model.train(data="./datasets/data.yaml",workers=0,epochs=130,batch=8,lr0=0.0005,optimizer="AdamW",weight_decay=0.0005,cos_lr=True,val=True)


#终端
#yolo task=detect mode=train model=runs/detect/train18/weights/best.pt data=yolov8_bvn.yaml epochs=150 batch=8 workers=0

#yolo cfg=default_copy.yaml
'''

from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 解决 OpenCV 与某些库的冲突
os.environ["ULTRALYTICS_OFFLINE"] = "1"       # 强制 Ultralytics 离线模式

model = YOLO("yolov8n.pt")
model.train(
    # ---------- 数据路径 ----------
    data="datasets/data.yaml",        # data.yaml文件路径
    
    # ---------- 训练轮次与批次 ----------
    epochs=200,                       # 训练轮数
    batch=8,                          # 批次大小
    imgsz=640,                        # 输入图像尺寸
    workers=0,                        # Windows下建议 0，避免多进程死锁

    # ---------- 优化器与学习率 ----------
    optimizer="AdamW",                # 优化器
    lr0=0.001,                        # 初始学习率
    weight_decay=0.0005,              # 权重衰减
    cos_lr=True,                      # 启用余弦退火，保持恒定学习率
    
    # ---------- 数据增强 ----------
    mosaic=1.0,                       # 马赛克增强
    mixup=0.2,                        # MixUp 增强
    copy_paste=0.0,                   # 复制粘贴
    flipud=0.2,                       # 上下翻转概率
    fliplr=0.5,                       # 左右翻转概率
    hsv_h=0.015,                      # 色调变化范围
    hsv_s=0.7,                        # 饱和度变化范围
    hsv_v=0.4,                        # 明度变化范围
    degrees=10,                       # 随机旋转角度
    translate=0.1,                    # 随机平移比例
    scale=0.5,                        # 随机缩放范围
    shear=0.0,                        # 剪切变换
    
    # ---------- 验证与早停 ----------
    val=True,                         # 是否启用验证集
    patience=20,                      # 早停耐心值
    
    # ---------- 硬件与输出 ----------
    device="cpu",                     # 使用 CPU
)

# ==================== 5. 训练完成提示 ====================
print("训练完成！")

