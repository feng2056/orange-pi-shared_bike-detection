# test_pytorch_model.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
from pathlib import Path


def test_pytorch_model():
    """测试PyTorch模型"""
    print("🧪 测试PyTorch模型 (best.pt)")
    print("=" * 50)

    # 1. 检查模型文件
    model_path = "best.pt"
    if not os.path.exists(model_path):
        # 尝试在训练目录中查找
        possible_paths = [
            "runs/detect/train2/weights/best.pt",
            "runs/train/shared_bicycle_detection/weights/best.pt",
            "best.pt"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

    if not os.path.exists(model_path):
        print(f"❌ 找不到PyTorch模型文件")
        print("请确保best.pt文件存在，或修改脚本中的路径")
        return None

    print(f"📦 加载PyTorch模型: {model_path}")

    # 2. 加载模型
    try:
        model = YOLO(model_path)
        print("✅ PyTorch模型加载成功")
    except Exception as e:
        print(f"❌ PyTorch模型加载失败: {e}")
        return None

    # 3. 测试图像
    test_images = []

    # 优先使用test.jpg
    if os.path.exists("test.jpg"):
        test_images.append("test.jpg")

    # 如果没有test.jpg，使用验证集图像
    if not test_images:
        val_dir = Path("dataset/images/val")
        if val_dir.exists():
            val_images = list(val_dir.glob("*.jpg"))[:3]
            test_images = [str(img) for img in val_images]

    if not test_images:
        print("❌ 找不到测试图像")
        return None

    results = {}

    for img_path in test_images:
        print(f"\n🔍 测试图像: {img_path}")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 无法读取图像: {img_path}")
            continue

        # 使用YOLO模型进行推理
        start_time = time.time()
        detections = model(img, conf=0.25, iou=0.45, imgsz=640, device='cpu')
        inference_time = time.time() - start_time

        # 解析结果
        boxes = []
        confidences = []

        for result in detections:
            if result.boxes is not None:
                for box in result.boxes:
                    # 获取坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()

                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(conf)

        # 保存结果图像
        result_img = img.copy()
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            color = (0, 255, 0)

            # 绘制框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_img, f'{conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 保存结果
        output_path = f"pytorch_result_{Path(img_path).stem}.jpg"
        cv2.imwrite(output_path, result_img)

        print(f"✅ 检测到 {len(boxes)} 个共享单车")
        print(f"📊 推理时间: {inference_time * 1000:.1f}ms")
        print(f"💾 结果保存到: {output_path}")

        results[img_path] = {
            'boxes': boxes,
            'confidences': confidences,
            'count': len(boxes),
            'time': inference_time,
            'output_path': output_path
        }

    return results


def print_pytorch_summary(results):
    """打印PyTorch模型测试总结"""
    if not results:
        return

    print("\n" + "=" * 50)
    print("📊 PyTorch模型测试总结")
    print("=" * 50)

    total_boxes = 0
    total_images = len(results)
    total_time = 0

    for img_path, result in results.items():
        boxes_count = result['count']
        inference_time = result['time']

        print(f"📷 {Path(img_path).name}:")
        print(f"  检测数量: {boxes_count}")
        print(f"  推理时间: {inference_time * 1000:.1f}ms")

        if boxes_count > 0:
            print(f"  置信度范围: {min(result['confidences']):.3f} - {max(result['confidences']):.3f}")

        total_boxes += boxes_count
        total_time += inference_time

    if total_images > 0:
        avg_boxes = total_boxes / total_images
        avg_time = total_time / total_images

        print(f"\n📈 总体统计:")
        print(f"  测试图像数: {total_images}")
        print(f"  总检测框数: {total_boxes}")
        print(f"  平均每张图像: {avg_boxes:.1f} 个")
        print(f"  平均推理时间: {avg_time * 1000:.1f}ms")


if __name__ == "__main__":
    pytorch_results = test_pytorch_model()
    print_pytorch_summary(pytorch_results)