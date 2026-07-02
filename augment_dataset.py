import os
import cv2
import albumentations as A
from pathlib import Path

# ================= 配置区域 =================
# 输入路径：原始数据集文件夹（里面应有 train/images 和 train/labels）
input_root = "./datasets"            # 你的原始数据目录（30张图片所在）
output_root = "./datasets_augmented" # 扩充后的输出目录

# 每张原图生成的变体数量（30张 * 3 = 90个变体 + 30张原图 = 120张）
augmentations_per_image = 3

# 原始的子文件夹（YOLO标准结构）
image_dir = "train/images"
label_dir = "train/labels"

# 输出子文件夹（保持相同结构）
output_image_dir = "train/images"
output_label_dir = "train/labels"

# 是否保留原始图片（True = 保留原图，总图数 = 原图 + 变体）
keep_original = True
# ===========================================

def get_augmentation_pipeline():
    """定义增强流水线（包含几何变换 + 色彩变换，同步更新边界框）"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def save_yolo_bboxes(filepath, bboxes, class_labels):
    """将边界框保存为YOLO格式文本"""
    with open(filepath, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            line = f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)

def main():
    # 创建输出目录
    output_img_path = Path(output_root) / output_image_dir
    output_lbl_path = Path(output_root) / output_label_dir
    output_img_path.mkdir(parents=True, exist_ok=True)
    output_lbl_path.mkdir(parents=True, exist_ok=True)

    input_img_path = Path(input_root) / image_dir
    input_lbl_path = Path(input_root) / label_dir

    # 获取所有图片文件
    img_files = [f for f in os.listdir(input_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(img_files)} 张原始图片")

    aug_pipeline = get_augmentation_pipeline()
    total_saved = 0

    for idx, img_file in enumerate(img_files):
        # 读取图片
        img_path = input_img_path / img_file
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"警告：无法读取 {img_path}，跳过")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取对应的YOLO标签文件
        label_file = Path(input_lbl_path) / (Path(img_file).stem + ".txt")
        bboxes = []
        class_labels = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        else:
            print(f"警告：标签文件 {label_file} 不存在，跳过该图片")
            continue

        if not bboxes:
            print(f"警告：{img_file} 没有有效边界框，跳过")
            continue

        # 1. 保存原始图片（如果保留）
        if keep_original:
            orig_img_out = output_img_path / img_file
            cv2.imwrite(str(orig_img_out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            save_yolo_bboxes(output_lbl_path / (Path(img_file).stem + ".txt"), bboxes, class_labels)
            total_saved += 1

        # 2. 生成多个增强变体
        for aug_idx in range(augmentations_per_image):
            try:
                augmented = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                if len(aug_bboxes) == 0:
                    continue

                stem = Path(img_file).stem
                ext = Path(img_file).suffix
                new_img_name = f"{stem}_aug{aug_idx+1}{ext}"
                new_img_path = output_img_path / new_img_name
                cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

                new_lbl_name = f"{stem}_aug{aug_idx+1}.txt"
                new_lbl_path = output_lbl_path / new_lbl_name
                save_yolo_bboxes(new_lbl_path, aug_bboxes, aug_labels)

                total_saved += 1
                print(f"已生成: {new_img_name} (框数: {len(aug_bboxes)})")
            except Exception as e:
                print(f"增强 {img_file} 的第 {aug_idx+1} 个变体时出错: {e}")

    print(f"\n✅ 完成！总共生成 {total_saved} 张图片（含原始图片）")
    print(f"扩充后的数据集位于: {output_root}")

if __name__ == "__main__":
    main()