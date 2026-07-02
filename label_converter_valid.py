import os
import re

# ========== 1. 数字-类别映射表 ==========
class_map = {
    # 类别0：蚯蚓
    9: 0, 29: 0, 56: 0, 87: 0, 102: 0,
    # 类别1：蚱蜢
    19: 1, 23: 1, 29: 1, 36: 1, 54: 1,195: 1,237: 1,
    # 类别2：蜗牛
    8: 2, 40: 2, 76: 2, 138: 2, 150: 2, 204: 2, 213: 2,
    # 类别3：黄蜂
    514: 3, 537: 3, 538: 3, 592: 3, 738: 3, 754: 3,
    # 类别4：甲壳虫
    37: 4, 41: 4, 134: 4, 169: 4, 234: 4, 240: 4
}

# ========== 2. 你的数据集路径 ==========
# 验证集路径
val_images = r"D:\ylj\A_data\机器学习\报告\机器学习课程设计报告\yolov8\datasets\valid\images"
val_labels = r"D:\ylj\A_data\机器学习\报告\机器学习课程设计报告\yolov8\datasets\valid\labels"

# 正则表达式：匹配文件名中-数字-的部分
pattern = re.compile(r'-(\d+)-')

def modify_labels(images_dir, labels_dir):
    print(f"开始处理: {images_dir}")
    count = 0
    skip_count = 0
    for img_name in os.listdir(images_dir):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # 从文件名中提取数字
        match = pattern.search(img_name)
        if not match:
            print(f"⚠️  跳过无数字的文件: {img_name}")
            skip_count += 1
            continue
        
        file_num = int(match.group(1))
        if file_num not in class_map:
            print(f"⚠️  跳过未定义数字的文件: {img_name} (数字: {file_num})")
            skip_count += 1
            continue
        
        new_class = class_map[file_num]
        # 找到对应的标签文件
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if not os.path.exists(label_path):
            print(f"⚠️  跳过无标签的文件: {img_name}")
            skip_count += 1
            continue
        
        # 修改标签文件中的类别号
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                # 把原来的类别号替换成新的编号
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts))
        
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        
        count += 1
    
    print(f"✅ 处理完成: 成功修改 {count} 个标签, 跳过 {skip_count} 个文件")

# 运行修改
modify_labels(val_images, val_labels)
print("\n🎉 所有标签修改完成！")